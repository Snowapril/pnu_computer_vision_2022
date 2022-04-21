import hw_utils as utils
import matplotlib.pyplot as plt


def main():
    # Test run matching with no ransac
    plt.figure(figsize=(20, 20))
    im = utils.Match('./data/scene', './data/box', ratio_thres=0.52)
    plt.title('Match')
    plt.imshow(im)
    im.save("scene-box-match.jpg")

    # Test run matching with ransac
    plt.figure(figsize=(20, 20))
    im = utils.MatchRANSAC(
        './data/library', './data/library2',
        ratio_thres=0.72, orient_agreement=5, scale_agreement=0.2)
    plt.title('MatchRANSAC')
    plt.imshow(im)
    im.save("library-match-ransac.jpg")

if __name__ == '__main__':
    main()
