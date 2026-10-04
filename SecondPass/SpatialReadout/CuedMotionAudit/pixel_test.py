"""Bounded CPU tests. Synthetic fixtures are tests, never audit data."""
import importlib.util
from pathlib import Path
import numpy as np

def test_known_translation():
    path = Path(__file__).with_name('pixel_observer.py')
    assert path.exists(), 'Pixel observer implementation is missing'
    spec = importlib.util.spec_from_file_location('pixel_observer', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rng = np.random.default_rng(27)
    a = np.zeros((27,27))
    a[tuple(rng.integers(4,22,(2,30)))] = 1
    for expected,(dx,dy) in enumerate(((1,0),(0,-1),(-1,0),(0,1))):
        b = np.roll(a,(dy,dx),(0,1))
        assert m.duration_scores(a,b).argmax() == expected

def test_subpixel_flow():
    import pixel_observer as m
    assert hasattr(m, 'patch_flow'), 'Subpixel flow implementation is missing'
    xy=np.array([[4.2,4.3],[12.7,5.8],[7.1,15.3]])
    def raster(p):
        a=np.zeros((23,23))
        for x,y in p:
            ix,iy=int(x),int(y)
            for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
                a[iy+dy,ix+dx]+=.48*((x-ix) if dx else (1-x+ix))*((y-iy) if dy else (1-y+iy))
        return a
    theta=.7
    delta=.375*np.array([np.cos(theta),np.sin(theta)])
    v=m.patch_flow(raster(xy),raster(xy+delta))
    assert len(v)==3
    assert np.allclose(v,delta,atol=1e-5)

if __name__ == '__main__':
    test_known_translation()
    test_subpixel_flow()
    print('pixel_test passed')
