import streamlit as st
from utils import detect_objects, get_image_bytes
import io



# ----------- Sidebar
page = st.sidebar.selectbox('Page Navigation', ["Objects Detector", "Text Generation", "other models (coming soon)"])

st.sidebar.markdown("""---""")
st.sidebar.write("Created by Maher https://github.com/Maherstad")
st.sidebar.image("assets/logo.png", width=100)



st.title('Object Detection')

sample_images = ['./assets/street.jpg', './assets/safari.jpg','./assets/bedroom.jpg']

selected_sample = st.selectbox('choose a sample image:', [''] + [i.split('/')[-1] for i in sample_images])

image_processed = False

if selected_sample:
    with open(f"./assets/{selected_sample}", "rb") as file:
        bytes_data = file.read()
    image_placeholder = st.empty()
    image_placeholder.image(bytes_data, caption='Selected Sample Image', use_column_width=True)
    image_processed = False
else:

    # Image upload
    uploaded_file = st.file_uploader("Or upload your own image ...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # To read file as bytes
        bytes_data = uploaded_file.getvalue()
        
        # Create a placeholder for the image
        image_placeholder = st.empty()
        
        # Display the uploaded image in the placeholder
        image_placeholder.image(bytes_data, caption='Original Image', use_column_width=100)
    
        # Create columns for buttons
col1, col2 = st.columns(2)
    

with col1:
    # Button to run object detection
    if st.button('Run Object Detection Model') and (selected_sample or uploaded_file):
        # Process the image
        with st.spinner(text = "Fetching model prediction"): #or 'Processing...'
            processed_image = detect_objects(bytes_data)
            img_byte_arr = io.BytesIO()
            processed_image.save(img_byte_arr, format='PNG')
            image_processed = True

        # Update the placeholder with the processed image
        image_placeholder.image(img_byte_arr.getvalue(), caption='Processed Image', use_column_width=True)

with col2:
    # Download button (enabled only after processing)
    if image_processed:
        processed_image_bytes = img_byte_arr.getvalue()
        st.download_button(
            label="Download Processed Image",
            data=processed_image_bytes,
            file_name="processed_image.png",
            mime="image/png"
        )
    else:
        # Show a disabled download button before processing
        st.download_button(
            label="Download Processed Image",
            data="",
            file_name="processed_image.png",
            mime="image/png",
            disabled=True
        )

