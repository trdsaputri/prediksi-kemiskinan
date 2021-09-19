from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)

model_file = open('finalized_model.pkl', 'rb')
model = pickle.load(model_file, encoding='bytes')

@app.route('/')
def index():
    return render_template('index.html', angka_kemiskinan=0)

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Prediksi angka kemiskinan berdasarkan inputan user
    '''
    pengeluaran_makanan_perkapita, Lapangan_kerja_informal_nonpertanian, rumah_milik, puas_ketersediaan_waktu_luang, puas_pekerjaan, puas_pendapatan, pengangguran_2 = [x for x in request.form.values()]

    data = []

    data.append(float(pengeluaran_makanan_perkapita))
    data.append(float(Lapangan_kerja_informal_nonpertanian))
    data.append(float(rumah_milik))
    data.append(float(puas_ketersediaan_waktu_luang))
    data.append(float(puas_pekerjaan))
    data.append(float(puas_pendapatan))
    data.append(float(pengangguran_2))

    new_data = np.array(data).reshape((1, -1))
    scaler = StandardScaler()
    
    scaled = scaler.fit_transform(new_data)

    prediction = model.predict(scaled)
    output = round(prediction[0], 2)

    return render_template('index.html', angka_kemiskinan=output)


if __name__ == '__main__':
    app.run(debug=True)