from ellipse_detector import EllipseDetector
from seal_comparator import SealComparator

if __name__ == "__main__":
    for ellipse in EllipseDetector()('images/source/test.png', 'images/output/'):
        SealComparator()(ellipse)