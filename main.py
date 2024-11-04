import os
from seg_fiber import Seger
from seg_fiber import segs2db

package_dir = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(package_dir,'seg_fiber/model/universal_tiny.pth')
CKPT_PATH = os.path.join(package_dir,'seg_fiber/model/universal_tiny.safetensors')

def seg_fiber(input_path:str, output_path:str, cuda_device_id:int=2,
              channel:int=0, bg_thres:int=200, chunk_size:int=300, splice:int=100000,
              roi:list=None):
    seger = Seger(ckpt_path=CKPT_PATH, bg_thres=bg_thres, cuda_device_id=cuda_device_id)
    segs = seger.process_whole(input_path, channel, chunk_size=chunk_size, splice=splice, roi=roi)
    segs2db(segs, output_path)
    
    return

if __name__ == '__main__':
    input_path = 'test/data/test.tif'
    output_path = 'test/out/test.db'
    roi = [0,0,0,300,300,300]
    roi = None
    seg_fiber(input_path, output_path, roi=roi)