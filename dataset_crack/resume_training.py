from ultralytics import YOLO

def main():
    model = YOLO(r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\dataset_crack\runs\detect\runs\pipe_proto\yolo26n_final_merged-7\weights\last.pt")
    model.train(resume=True)

if __name__ == "__main__":
    main()