from rvfn.utils.pointcloud import show_pointcloud
import argparse 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ply_file', help='input pointcloud data')
    args = parser.parse_args()

    show_pointcloud(args.ply_file)

