import numpy as np
import pickle

# 9個模型路徑
model_paths = [
    'models/anger_model.pkl',
    'models/forget_model.pkl',
    'models/emptiness_model.pkl',
    'models/hopelessness_model.pkl',
    'models/loneliness_model.pkl',
    'models/sadness_model.pkl',
    'models/suicide_model.pkl',
    'models/worthlessness_model.pkl',
    'models/NoDepressionEmo_model.pkl'
]

models = []

def load_models():
    global models
    models = []
    for path in model_paths:
        with open(path, 'rb') as f:
            models.append(pickle.load(f))

load_models()

def predict_with_ensemble(X):
    # X是輸入特徵的DataFrame或array，shape=(1, n_features)

    # 預測所有模型，output shape = (9,)
    preds = []
    for model in models:
        pred = model.predict(X)  # 預測 0 或 1，shape=(1,)
        preds.append(pred[0])
    preds = np.array(preds)  # shape=(9,)

    # 取得 NoDepressionEmo 預測結果（最後一個）
    no_depression_pred = preds[-1]

    if no_depression_pred == 1:
        # 無情緒，前8個設0，最後保持1
        adjusted_preds = np.array([0]*8 + [1])
    else:
        # 有情緒，保留前8個，最後設0
        adjusted_preds = np.array(list(preds[:-1]) + [0])

    # 將結果轉成字串，標示有標籤的情緒
    emotion_list = ['anger', 'brain dysfunction (forget)', 'emptiness', 'hopelessness',
                    'loneliness', 'sadness', 'suicide intent', 'worthlessness', 'NoDepressionEmo']

    detected_emotions = [emotion for emo_flag, emotion in zip(adjusted_preds, emotion_list) if emo_flag == 1]

    if not detected_emotions:
        return "預測結果：無明確情緒"
    else:
        return "預測結果：" + ", ".join(detected_emotions)
