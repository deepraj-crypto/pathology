import streamlit as st
import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from google.oauth2 import service_account
from gsheetsdb import connect
import gspread

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
    ],
)
conn = connect(credentials=credentials)

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('./model_insv3-13-0.7590.hdf5', compile=False)
  model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
  )
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Colorectal Polyps Classification
         """
         )

col1,col2=st.columns(2)

with col1:
  file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
with col2:
  patient=st.text_input('Patient ID:')
  doctor_prediction=st.text_input("Doctor's Prediction:")
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):

        size = (299,299)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.

        img_reshape = img[np.newaxis,...]

        prediction = model.predict(img_reshape)

        return prediction

def save_to_excel(name, image_path, prediction, file_path, doctor_prediction):
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
        ],
    )
    conn = connect(credentials=credentials)
    client = gspread.authorize(credentials)
    sheet = client.open_by_url(file_path)
    worksheet = sheet.get_worksheet(0)
    new_row = [name, image_path, prediction, doctor_prediction]
    worksheet.append_row(new_row)
    # df = pd.DataFrame({'Name': [name], 'Image Path': [image_path], 'Prediction': [prediction]})
    # excel_file=pd.ExcelFile(file_path)
    # if 'Sheet1' in excel_file.sheet_names:
     # with pd.ExcelWriter(file_path, mode='a',if_sheet_exists="overlay") as writer:
      #    df.to_excel(writer, sheet_name='Sheet1', index=False)

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names=['HP', 'NORM', 'TA.HG']
    score = tf.nn.softmax(predictions[0])
    # st.write(predictions)
    # st.write(score)
    if np.max(score) < 0.50:
        string = "This is not an image of pathology, please enter a valid image"
    elif np.max(score) >= 0.50:
        string="This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    st.success(string)

    if st.button('Save to Excel'):
        if patient == '':
            st.error('Please enter a patient name')
        elif doctor_prediction == '':
            st.error('Please enter your prediction')
        elif np.max(score) < 0.50:
            st.error('This is not an image of pathology, please enter a valid image')
        else:
            sheet_url = st.secrets["private_gsheets_url"]
            save_to_excel(patient, file.name, string, sheet_url, doctor_prediction)
            st.success('Data saved to Excel')
