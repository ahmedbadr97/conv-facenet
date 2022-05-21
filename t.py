import deepface
detector_backend="retinaface"
target_size=(225,225)
raw_data_path="dataset/raw/img_celeba"
prep_data_path="dataset/preprocessed/img_celeb"
imgs_list=os.listdir(raw_data_path)
total_img=len(imgs_list)
cnt=0
time_sum=0
for img_name in imgs_list:
    img_process_start=time.time()
    img=preprocess_face(f"{raw_data_path}/{img_name}",target_size=target_size,detector_backend=detector_backend)
    cv2.imwrite(img,f"{prep_data_path}/{img_name}")
    cnt += 1.0
    finished = int((cnt * 10) / total_img)
    remaining = 10 - finished
    img_process_end = time.time()
    time_sum += (img_process_end - img_process_start)
    avg_time = time_sum / cnt
    time_remaing = avg_time * (total_img - cnt)
    sys.stdout.write("\r Data processing  [" + str(
                "=" * int((cnt * 10) / total_img) + str("." * remaining) + "] time remaining = " + str(
                    time_remaing / 60.0)[:8]))