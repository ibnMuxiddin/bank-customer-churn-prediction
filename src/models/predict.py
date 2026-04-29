import pandas as pd
import joblib
from features.add_attr import FeatureGenerator
from data.load import load_data

def test_model():
    # 1. Modelni yuklaymiz
    model_path = "../../models/best_svm_pipeline.pkl" 
    
    try:
        loaded_pipeline = joblib.load(model_path)
        print("Model muvaffaqiyatli yuklandi!\n")
    except FileNotFoundError:
        print(f"Xatolik: Model topilmadi. Manzilni tekshiring: {model_path}")
        return

    # 2. Test qilish uchun xom ma'lumotlar (DataFrame)
    df = load_data("Churn_Modelling.xls")
    df = df.drop("Exited", axis=1)
    test = df.sample(100)

    print("--- Test Ma'lumotlari ---")
    print(test.head(10))
    print("\n" + "="*50 + "\n")

    # 3. Bashorat qilish (Predict va Predict_Proba)
    predictions = loaded_pipeline.predict(test)
    
    # Agar modelda predict_proba bo'lsa (SVM da probability=True bo'lishi kerak)
    try:
        probabilities = loaded_pipeline.predict_proba(test)[:, 1]
    except AttributeError:
        probabilities = ["Mavjud emas"] * len(predictions)

    # 4. Natijalarni chiroyli ko'rinishda chiqarish
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        status = "Ketadi (Churn)" if pred == 1 else "Qoladi (Not Churn)"
        prob_text = f"{prob:.1%}" if isinstance(prob, float) else prob
        
        print(f"Mijoz {i+1}:")
        print(f"  Bashorat: {status}")
        print(f"  Ketish ehtimolligi: {prob_text}")
        print("-" * 30)

if __name__ == "__main__":
    test_model()