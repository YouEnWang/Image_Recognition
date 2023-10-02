### Integration of MediaPipe and SIFT Technologies for Accurate Exercise Posture Recognition
## 內容簡介
本研究旨在研究SIFT演算法，並運用其辨識並確定運動員在運動過程中的姿勢是否有任何變化。不正確的運動姿勢是運動相關傷害的主要原因，我們使用Google MediaPipe對影像進行預處理，接著運用SIFT演算法在預處理後的圖像上提取特徵點和計算特徵描述符，並分析結果計算出動作相似性，以識別運動中的姿勢差異。從實驗結果中我們觀察到，當動作不正常時，相似性存在顯著差異。我們計劃在未來進一步發展我們的研究，通過觀察量化的相似性測量結果，使用者可以在運動中適當地調整身體部位，以降低運動傷害發生的機率。

## 貢獻程度(組內共有三人)
王佑恩 (40%)：論文內容整合、Python OpenCV程式設計、整合MediaPipe以及SIFT演算法、研究方法設計、研究參數測試、辨識結果分析

## 研究心得
我們設計的程式模型有確實達到預設的研究目標，研究過程也發現，整合Google MediaPipe Pose以及SIFT演算法，可以達成高精度和高可信度的姿勢估計與特徵匹配。我覺得待改進的其中一點是，此次在設計模型時是直接使用Google MediaPipe套件，沒有對其演算法有較深入的研究。若延續此研究主題的話，我會朝影像辨識所需之機器學習演算法方向進行深入研究，並應用於嵌入式系統領域之研究主題。

## 其他說明
論文投稿至iFUZZY 2023 International Conference on Fuzzy Theory and Its Applications獲得錄取
