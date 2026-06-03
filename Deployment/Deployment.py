from flask import Flask, request, jsonify
import joblib
import numpy as np

# نعمل التطبيق
app = Flask(__name__)

# نحمل الموديل
model = joblib.load("cancer_model.pkl")

# نعمل route للتنبؤ
@app.route('/predict', methods=['POST'])
def predict():
    # استلام البيانات من JSON
    data = request.get_json()

    # تحويل القيم لمصفوفة (array)
    input_data = np.array([list(data.values())])

    # التنبؤ
    prediction = model.predict(input_data)[0]

    # إرجاع النتيجة
    return jsonify({
        "diagnosis": "Malignant" if prediction == 'M' else "Benign"
    })

# تشغيل السيرفر
if __name__ == '__main__':
    app.run(debug=True)