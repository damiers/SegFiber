import sqlite3
from pathlib import Path

from tqdm import tqdm

from .neurodb_sqlite import NeurodbSQLite


def merge_z_slab_databases(input_dir, output, reset=False):
    input_dir = Path(input_dir).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    sources = sorted(
        path for path in input_dir.glob("*.db") if path.resolve() != output
    )
    if not sources:
        raise FileNotFoundError(f"No .db files found in {input_dir}")
    if output.exists() and not reset:
        raise FileExistsError(f"Output already exists: {output}")
    if output.exists():
        output.unlink()

    output.parent.mkdir(parents=True, exist_ok=True)
    NeurodbSQLite(output)
    expected = [0, 0, 0]

    connection = sqlite3.connect(output, uri=True)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        progress = tqdm(
            sources,
            desc="Merging z-slab DBs",
            unit="db",
            dynamic_ncols=True,
        )
        for source in progress:
            progress.set_postfix_str(source.name)
            sid_offset = connection.execute(
                "SELECT COALESCE(MAX(sid), 0) FROM segs"
            ).fetchone()[0]
            nid_offset = connection.execute(
                "SELECT COALESCE(MAX(nid), 0) FROM nodes"
            ).fetchone()[0]
            connection.execute(
                "ATTACH DATABASE ? AS slab",
                (f"{source.as_uri()}?mode=ro",),
            )
            try:
                if connection.execute(
                    "SELECT EXISTS(SELECT 1 FROM slab.actions)"
                ).fetchone()[0]:
                    raise ValueError(f"Input database contains actions: {source}")
                counts = [
                    connection.execute(
                        f"SELECT COUNT(*) FROM slab.{table}"
                    ).fetchone()[0]
                    for table in ("segs", "nodes", "edges")
                ]
                connection.execute("BEGIN")
                connection.execute(
                    """
                    INSERT INTO segs (sid, points, version, date)
                    SELECT sid + ?, points, version, date FROM slab.segs
                    """,
                    (sid_offset,),
                )
                connection.execute(
                    """
                    INSERT INTO nodes
                        (nid, x, y, z, creator, type, checked, status, sid, cid, date)
                    SELECT
                        nid + ?, x, y, z, creator, type, checked, status,
                        CASE WHEN sid > 0 THEN sid + ? ELSE sid END,
                        CASE WHEN cid > 0 THEN cid + ? ELSE cid END,
                        date
                    FROM slab.nodes
                    """,
                    (nid_offset, sid_offset, sid_offset),
                )
                connection.execute(
                    """
                    INSERT INTO edges (src, dst, creator, date)
                    SELECT src + ?, dst + ?, creator, date FROM slab.edges
                    """,
                    (nid_offset, nid_offset),
                )
                connection.commit()
                expected = [total + count for total, count in zip(expected, counts)]
                progress.set_postfix_str(
                    f"{source.name} | segs={expected[0]} nodes={expected[1]}",
                    refresh=False,
                )
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.execute("DETACH DATABASE slab")
        _validate_merge(connection, expected)
    finally:
        connection.close()
    return output


def _validate_merge(connection, expected):
    actual = [
        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("segs", "nodes", "edges")
    ]
    if actual != expected:
        raise RuntimeError(f"Merged row counts do not match: {actual} != {expected}")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise RuntimeError("Merged database contains invalid foreign keys")
    rtree_count = connection.execute("SELECT COUNT(*) FROM nodes_rtree").fetchone()[0]
    if rtree_count != actual[1]:
        raise RuntimeError("Merged database spatial index is incomplete")
