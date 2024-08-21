from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# تحميل النموذج المدرب من الملف
with open('knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # الحصول على البيانات من النموذج (Form) الذي يتم إرساله من الصفحة الرئيسية

    if request.method == 'POST':
        try:
            # الحصول على القيم من request.form
            input_values = []
            for key, value in request.form.items():
                input_values.append(float(value))

            # تحويل القيم إلى مصفوفة numpy
            input_data = np.array(input_values).reshape(1, -1)

            # إجراء التوقع باستخدام النموذج
            prediction = model.predict(input_data)[0]

            # إرجاع النتيجة لعرضها في صفحة الويب
            return render_template('index.html', prediction_text=f'النتيجة المتوقعة: {prediction}')
        except ValueError:
            return render_template('index.html', prediction_text="يرجى إدخال أرقام صحيحة.")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
