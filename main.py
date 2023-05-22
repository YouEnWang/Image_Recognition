# 用SIFT辨識mediapipe做完姿勢辨識之骨架的特徵點

import cv2
import mediapipe as mp
import numpy as np
import math

# https://google.github.io/mediapipe/

# 計算角度
def getAngle(firstPoint, midPoint, lastPoint):
    result = math.degrees(math.atan2(lastPoint.y - midPoint.y, lastPoint.x - midPoint.x)
                           - math.atan2(firstPoint.y - midPoint.y, firstPoint.x - midPoint.x))
    result = abs(result)    # 角度恆正
    if result > 180:
        result = 360.0 - result
    return result

# 使用MediaPipe庫中的姿勢估測模型
mpDraw = mp.solutions.drawing_utils                     # MediaPipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles         # MediaPipe 繪圖樣式
mpPose = mp.solutions.pose                              # MediaPipe 偵測姿勢方法
pose = mpPose.Pose()

# 創建 SIFT 物件
sigma = 1.6
sift = cv2.SIFT_create(0, 5, 0.09, 10, sigma)

# 創建 Flann Matcher 物件，用於特徵匹配
flann = cv2.FlannBasedMatcher()

#建立FLANN匹配對象
flann_params = dict(algorithm = 6, 
                    table_number = 6,
                    key_size = 12,
                    multi_probe_level = 1)

# # 讀取影片檔案(為了先行計算相似度)
# cap1_1 = cv2.VideoCapture('data/2-1.mp4')
# cap2_1 = cv2.VideoCapture('data/2-3.mp4')

# # 記錄所有match ratio
# total_match_ratio = sim.count_Similarity.get_Similarity(cap1_1, cap2_1)
# counter = 0

# 讀取影片檔案
cap1 = cv2.VideoCapture('data/2-1.mp4')
cap2 = cv2.VideoCapture('data/2-3.mp4')
cap3 = cv2.VideoCapture('data/2-2.mp4')

# 取得影片的總幀數
total_frame1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
total_frame2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
total_frame3 = int(cap3.get(cv2.CAP_PROP_FRAME_COUNT))

# 取得影片的幀率(FPS)
FPS1 = int(cap1.get(cv2.CAP_PROP_FPS))
FPS2 = int(cap2.get(cv2.CAP_PROP_FPS))
FPS3 = int(cap3.get(cv2.CAP_PROP_FPS))

# 計算基準影片的總時長
duration_video1 = total_frame1 / FPS1
duration_video3 = total_frame3 / FPS3

# 將時間存下
save_time = []

# 原先用來記錄所有match ratio(未經mapping)
total_match_ratio = []

# 紀錄每幀的內點群數量
total_inlier = []

# 紀錄keypoint個數
sum_kp1 = []
sum_kp2 = []

with mpPose.Pose(
    min_detection_confidence = 0.5,
    enable_segmentation = True,     # 額外設定 enable_segmentation 參數
    min_tracking_confidence = 0.5) as pose:

    while True:
        # 相似度計算所需
        matchNum = 0
        match_ratio = []

        # 讀取一幀影像
        success1, img1 = cap1.read()
        success2, img2 = cap2.read()

        if not success1 or not success2:
            print('Ignoring empty image frame.')
            break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        img1.flags.writeable = False
        img2.flags.writeable = False

        img1 = cv2.resize(img1, (700, 500))
        img2 = cv2.resize(img2, (700, 500))

        # 將圖像複製，用以將結果呈現於圖像上
        img3 = img1.copy()
        img4 = img2.copy()

        # 保留原始圖像以利後續運用
        img_original_1 = img1.copy()
        img_original_2 = img2.copy()

        # 建立空白的背景
        shape = (500, 700, 3)
        background = np.full(shape, 255).astype(np.uint8)
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        bg1 = background.copy()
        bg2 = background.copy()
        bg3 = background.copy()
        bg4 = background.copy()

        # 將影像轉換為RGB格式，因為MediaPipe庫中的姿勢估測模型需要RGB格式的影像
        imgRGB1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        imgRGB2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # 使用MediaPipe庫的姿勢估測模型來檢測影片中的人體姿勢，並返回關鍵點的位置
        results1 = pose.process(imgRGB1)
        results2 = pose.process(imgRGB2)
        # print(results.pose_landmarks)

        try:
            # 使用 try 避免抓不到姿勢時發生錯誤
            condition1 = np.stack((results1.segmentation_mask,) * 3, axis=-1) > 0.1
            condition2 = np.stack((results2.segmentation_mask,) * 3, axis=-1) > 0.1

            # 如果滿足模型判斷條件 ( 表示要換成背景 )，回傳 True
            img1 = np.where(condition1, img1, bg1)
            img2 = np.where(condition2, img2, bg2)
            # 將主體與背景合成，如果滿足背景條件，就更換為bg 的像素，不然維持原本img 的像素
        except:
            pass

        # 如果檢測到了人體姿勢，則在原始影像上繪製關鍵點
        if results1.pose_landmarks and results2.pose_landmarks:
            img1.flags.writeable = True
            img2.flags.writeable = True

            # 建立空的array來儲存骨架
            landmark_list1 = []
            landmark_list2 = []

            # 繪製關鍵點和連接線，將其繪製在img1和img2變量上
            mpDraw.draw_landmarks(img1, 
                                  results1.pose_landmarks, 
                                  mpPose.POSE_CONNECTIONS,
                                  landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style())
            mpDraw.draw_landmarks(img2, 
                                  results2.pose_landmarks, 
                                  mpPose.POSE_CONNECTIONS,
                                  landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style())

            # 對於每個關鍵點，在原始影像上畫一個小圓點
            for id, lm in enumerate(results1.pose_landmarks.landmark):
                h, w, c = img1.shape
                print(id, lm)
        
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                landmark_list1.append([cx, cy])
                cv2.circle(img1, (cx, cy), 3, (255, 0, 0), cv2.FILLED)

            for id, lm in enumerate(results2.pose_landmarks.landmark):
                h, w, c = img2.shape
                print(id, lm)
                            
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list2.append([cx, cy])
                cv2.circle(img2, (cx, cy), 3, (255, 0, 0), cv2.FILLED)

            # 將骨架放在空白背景上
            for pair in mpPose.POSE_CONNECTIONS:
                cv2.line(bg3, landmark_list1[pair[0]], landmark_list1[pair[1]], (0, 255, 0), 3)
                cv2.line(bg4, landmark_list2[pair[0]], landmark_list2[pair[1]], (0, 255, 0), 3)

            # 顯示角度
            right_shoulder1 = results1.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_SHOULDER]
            right_hip1 = results1.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_HIP]
            right_knee1 = results1.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_KNEE]

            rightHipAngle1 = getAngle(right_shoulder1, right_hip1, right_knee1)
            print('rightHipAngle1', rightHipAngle1)

            right_shoulder2 = results2.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_SHOULDER]
            right_hip2 = results2.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_HIP]
            right_knee2 = results2.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_KNEE]

            rightHipAngle2 = getAngle(right_shoulder2, right_hip2, right_knee2)
            print('rightHipAngle2', rightHipAngle2)
            
            # # 操作SIFT
            # 將影像轉換為灰度圖像
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # 進行 SIFT 特徵提取
            kp1, des1 = sift.detectAndCompute(gray1, None)
            kp2, des2 = sift.detectAndCompute(gray2, None)

            # 紀錄keypoint個數
            sum_kp1.append(len(kp1))
            sum_kp2.append(len(kp2))
            
            # 匹配描述符
            matches = flann.knnMatch(des1, des2, 2)

            # 篩選最佳之匹配的描述符
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            
            # 對good以歐氏距離進行排序
            good = sorted(good, key = lambda x:x.distance)

            # RANSAC操作
            if len(good) > 10:
                # 獲取匹配點的位置
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                # 進行透視變換估計
                ransacReprojThreshold = 5.0
                M, mask = cv2.findHomography(src_pts,
                                             dst_pts, 
                                             cv2.RANSAC, 
                                             ransacReprojThreshold, 
                                             maxIters = 6000)
                
                # .ravel(): 將多維矩陣轉換為一維矩陣
                # .tolist(): 將數據化為list
                matchesMask = mask.ravel().tolist()

                # 取得內點群數量
                inlier_count = matchesMask.count(1)
                total_inlier.append(inlier_count)

            else:
                print('Not enough matches are found - %d%d' % (len(good), 10))
                matchesMask = 0     # 處理good不夠的狀況
            
            # 畫出匹配的線與特徵點
            draw_params = dict(matchColor = (0,255,0), 
                            singlePointColor = (0,0,255),
                            matchesMask = matchesMask,
                            flags = 4) 
            # flags=2: draw only inliers
            # flags=4: draw the circle around keypoint
            
            # 相似度計算
            matchNum = np.sum(matchesMask)
            if len(matches) > 0:
                matchRatio = (matchNum/len(matches))*100
            
            match_ratio.append(matchRatio)
            total_match_ratio.append(matchRatio)

            # 繪製匹配結果
            # 用if statement避開匹配點不夠的狀況
            if np.sum(matchesMask) > 0:
                # 相似度過低之處理
                danger_value = 15
                if matchRatio < danger_value:
                    bad_kp1 = []
                    bad_kp2 = []
                    matches_list = []
                    for m, n in matches:
                        matches_list.append(m)

                    # 此項為0
                    for i, kp in enumerate(kp1):
                        if i not in [m.queryIdx for m in good]:
                            bad_kp1.append(kp)

                    for i, kp in enumerate(kp2):
                        if i not in [m.trainIdx for m in good]:
                            bad_kp2.append(kp)

                    # 要以tuple型態存
                    bad_kp1 = tuple(bad_kp1)
                    bad_kp2 = tuple(bad_kp2)

                    # 將目標影像轉為紅色
                    img4[:,:,0] = 0      # 圖片藍色元素歸0
                    img4[:,:,1] = 0      # 圖片綠色元素歸0

                    img = np.hstack((cv2.drawKeypoints(img3, bad_kp1, img3, color=(255, 0, 0)), cv2.drawKeypoints(img4, bad_kp2, img4, color=(255, 0, 0))))                    

                    
                else:
                    img = cv2.drawMatches(img3, kp1, img4, kp2, good, None, **draw_params)

                # 在視窗指定位置放上文字
                # cv2.putText(img, f'Similarity: {float(round(total_match_ratio[counter], 2))}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2)
                cv2.putText(img, f'Similarity: {float(round(matchRatio, 2))}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2)
                # cv2.putText(img, f'Right Hip Angle: {float(round(rightHipAngle1, 2))}', (20,470), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2)
                # cv2.putText(img, f'Right Hip Angle: {float(round(rightHipAngle2, 2))}', (720,470), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2)

                
            
            else:       # 相似度為0時，因為沒有匹配點，所以只顯示關鍵點
                # 將目標影像轉為紅色
                img4[:,:,0] = 0      # 圖片藍色元素歸0
                img4[:,:,1] = 0      # 圖片綠色元素歸0

                img = np.hstack((cv2.drawKeypoints(img3, kp1, img3, color=(255, 0, 0)), cv2.drawKeypoints(img4, kp2, img4, color=(255, 0, 0))))
                
                # 在視窗指定位置放上文字
                # cv2.putText(img, f'Similarity: {float(round(total_match_ratio[counter], 2))}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2)
                cv2.putText(img, f'Similarity: {float(round(matchRatio, 2))}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2)
                # cv2.putText(img, f'Right Hip Angle: {float(round(rightHipAngle1, 2))}', (20,470), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2)
                # cv2.putText(img, f'Right Hip Angle: {float(round(rightHipAngle2, 2))}', (720,470), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2)

        # counter += 1
        
        # 繪製結果(書面論文顯示)
        img_original = np.hstack((img_original_1, img_original_2))
        img_RGB = np.hstack((imgRGB1, imgRGB2))
        img_skeleton = np.hstack((bg3, bg4))
        img_new = np.hstack((img1, img2))
        img_gray = np.hstack((gray1, gray2))
        # img_without_background = np.hstack()      # 顯示去背的人像

        # # 顯示秒數(以右邊影片為主)
        # 取得當前幀數
        current_frame1 = int(cap1.get(cv2.CAP_PROP_POS_FRAMES))
        current_frame2 = int(cap2.get(cv2.CAP_PROP_POS_FRAMES))
        
        # 計算當前時間
        current_time1 = current_frame1/FPS1
        current_time2 = duration_video1 + duration_video3 + (current_frame2/FPS2)
        save_time.append(current_time2)
        # 在視窗中顯示當前時間
        cv2.putText(img, f"Time: {current_time1:.2f}s", (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, f"Time: {current_time2:.2f}s", (720, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow('img', img)
        # cv2.imshow('img_skeleton', img_skeleton)
        # cv2.imshow('img_new', img_new)
        # cv2.imshow('img_original', img_original)
        # cv2.imshow('img_RGB', img_RGB)
        # cv2.imshow('img_gray', img_gray)

        # cv2.imshow('img1', img1)
        # cv2.imshow('sobelxy1', sobelxy1)
        # cv2.imshow('sobelxy2', sobelxy2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # 按下中字的空白鍵暫停；再按一次(任意鍵)則繼續執行
        if cv2.waitKey(30) == 32:
            cv2.waitKey(0)


# 釋放攝影機
cap1.release()
cap2.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
