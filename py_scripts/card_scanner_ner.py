import json
import os
import sys
import cv2
import numpy as np
import pytesseract
import pandas as pd
import spacy
import re
import string
import warnings
# Ignore all warnings
warnings.filterwarnings('ignore')
#warnings.filterwarnings("ignore", category=UserWarning, module="spacy")
#warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

# Constants and settings
BASE_DIR = os.getcwd()
MEDIA_DIR = os.path.join(BASE_DIR, "media")
UPLOAD_IMAGE_PATH = os.path.join(MEDIA_DIR, "upload.jpg")
RESIZE_IMAGE_PATH = os.path.join(MEDIA_DIR, "resize_image.jpg")
WRAP_IMAGE_PATH = os.path.join(MEDIA_DIR, "magic_color.jpg")
BB_IMAGE_PATH = os.path.join(MEDIA_DIR, "bounding_box.jpg")

# Ensure media directory exists
os.makedirs(MEDIA_DIR, exist_ok=True)

# Helper functions
def save_image(filepath, image):
    """Save an image to the specified filepath."""
    cv2.imwrite(filepath, image)

def clean_text(txt):
    """Clean text by removing whitespace and punctuation."""
    whitespace = string.whitespace
    punctuation = "!#$%&\'()*+:;<=>?[\\]^`{|}~"
    table_whitespace = str.maketrans('', '', whitespace)
    table_punctuation = str.maketrans('', '', punctuation)
    text = str(txt)
    removewhitespace = text.translate(table_whitespace)
    removepunctuation = removewhitespace.translate(table_punctuation)
    return str(removepunctuation)

def parse_text(text, label):
    """Parse text based on its label for specific cleaning."""
    if label == 'PHONE':
        # Remove all non-digit characters
        text = re.sub(r'\D', '', text)
    elif label == 'EMAIL':
        # Refined regex to match valid email addresses
        text = re.sub(r'[^A-Za-z0-9@_.\-]', '', text)
    elif label == 'WEB':
        # Refined regex for web URLs
        text = re.sub(r'[^A-Za-z0-9:/.%#\-]', '', text)
    elif label == 'NAME':
        # Allow some special characters like apostrophes and hyphens
        text = re.sub(r"[^a-zA-Z'\- ]", '', text).title()
    elif label == 'DES':
        # Allow some special characters like apostrophes and hyphens in descriptions
        text = re.sub(r"[^a-zA-Z'\- ]", '', text).title()
    elif label == 'ORG':
        # Allow special characters like periods, ampersands, and hyphens in organization names
        text = re.sub(r"[^a-zA-Z0-9'\-\.& ]", '', text).title()
    return text



# group the label
class groupgen():
    def __init__(self):
        self.id = 0
        self.text = ''
        
    def getgroup(self,text):
        if self.text == text:
            return self.id
        else:
            self.id +=1
            self.text = text
            return self.id

# Initializing the groupgen object
grp_gen = groupgen()

class DocumentScanner:
    """Class for scanning and processing documents."""
    def __init__(self):
        pass

    @staticmethod
    def resizer(image, width=500):
        """Resize the image while maintaining the aspect ratio."""
        h, w, c = image.shape
        height = int((h / w) * width)
        size = (width, height)
        resized_image = cv2.resize(image, (width, height))
        return resized_image, size

    @staticmethod
    def apply_brightness_contrast(input_img, brightness=0, contrast=0):
        """Adjust the brightness and contrast of an image."""
        buf = input_img.copy()
        if brightness != 0:
            shadow = max(brightness, 0)
            highlight = 255 + min(brightness, 0)
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow
            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)

        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

    def scan_document(self, image_path):
        """Process an image to extract and enhance the document region."""
        image = cv2.imread(image_path)
        resized_image, size = self.resizer(image)
        save_image(RESIZE_IMAGE_PATH, resized_image)

        try:
            detail = cv2.detailEnhance(resized_image, sigma_s=20, sigma_r=0.15)
            # Apply brightness and contrast adjustment
            magic_color = self.apply_brightness_contrast(detail, brightness=40, contrast=60)
            save_image(WRAP_IMAGE_PATH, magic_color)

            return magic_color

        except Exception as e:
            print(f"Error during document scanning: {e}")

        return None

class NERPrediction:
    """Class for Named Entity Recognition (NER) predictions."""
    def __init__(self, model_path):
        """Initialize the NER model."""
        self.model = spacy.load(model_path)

    def get_predictions(self, image):
        """Extract text from an image and run NER predictions."""
        tess_data = pytesseract.image_to_data(image)
        tess_list = [row.split('\t') for row in tess_data.split('\n')]
        df = pd.DataFrame(tess_list[1:], columns=tess_list[0]).dropna()
        df['text'] = df['text'].apply(clean_text)

        df_clean = df.query('text != "" ')
        content = " ".join(df_clean['text'].tolist())

        doc = self.model(content)
        # converting doc in json
        docjson = doc.to_json()
        doc_text = docjson['text']

        # creating tokens
        datafram_tokens = pd.DataFrame(docjson['tokens'])
        datafram_tokens['token'] = datafram_tokens[['start', 'end']].apply(
            lambda x: doc_text[x[0]:x[1]], axis=1)

        right_table = pd.DataFrame(docjson['ents'])[['start', 'label']]
        datafram_tokens = pd.merge(datafram_tokens, right_table, how='left', on='start')
        datafram_tokens.fillna('O', inplace=True)

        # Join label to dataframe
        df_clean.loc[:, 'end'] = df_clean['text'].apply(lambda x: len(x)+1).cumsum() - 1
        df_clean.loc[:, 'start'] = df_clean[['text', 'end']].apply(lambda x: x[1] - len(x[0]), axis=1)


        # Inner join with start 
        dataframe_info = pd.merge(df_clean, datafram_tokens[['start', 'token', 'label']], how='inner', on='start')
        # Bounding Box Processing
        bb_df = dataframe_info.query("label != 'O' ")

        bb_df.loc[:, 'label'] = bb_df['label'].apply(lambda x: x[2:])
        bb_df.loc[:, 'group'] = bb_df['label'].apply(grp_gen.getgroup)

        # Right and bottom of bounding box
        bb_df.loc[:, ['left', 'top', 'width', 'height']] = bb_df[['left', 'top', 'width', 'height']].astype(int)
        bb_df.loc[:, 'right'] = bb_df['left'] + bb_df['width']
        bb_df.loc[:, 'bottom'] = bb_df['top'] + bb_df['height']

        # Tagging: Group by entity group
        col_group = ['left', 'top', 'right', 'bottom', 'label', 'token', 'group']
        group_tag_img = bb_df[col_group].groupby(by='group')
        img_tagging = group_tag_img.agg({
            'left': 'min',
            'right': 'max',
            'top': 'min',
            'bottom': 'max',
            'label': np.unique,
            'token': lambda x: " ".join(x)
        })
        img_bb = image.copy()
        for l, r, t, b, label, token in img_tagging.values:
            cv2.rectangle(img_bb, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(img_bb, str(label), (l, t), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

        # Entity extraction
        info_array = dataframe_info[['token', 'label']].values
        entities = dict(NAME=[], ORG=[], DES=[], PHONE=[], EMAIL=[], WEB=[])
        previous = 'O'

        for token, label in info_array:
            bio_tag = label[0]
            label_tag = label[2:]

            # Parse the token based on its label
            text = parse_text(token, label_tag)
            if bio_tag in ('B', 'I'):
                if previous != label_tag:
                    entities[label_tag].append(text)
                else:
                    if bio_tag == "B":
                        entities[label_tag].append(text)
                    else:
                        if label_tag in ("NAME", 'ORG', 'DES'):
                            entities[label_tag][-1] = entities[label_tag][-1] + " " + text
                        else:
                            entities[label_tag][-1] = entities[label_tag][-1] + text
            previous = label_tag

        return img_bb, entities
        '''
        results = {"entities": []}

        for ent in doc.ents:
            label = ent.label_
            text = parse_text(ent.text, label)
            results["entities"].append({"label": label, "text": text})

        return results
        '''

def main(input_image_path):
    """Main function to scan a document and predict NER entities."""
    scanner = DocumentScanner()
    predictor = NERPrediction(model_path="./models/model-best/")
    # Scan the input image
    processed_image = scanner.scan_document(input_image_path)
    if processed_image is not None:
        #print("Document scanned successfully.")
        bb_image, predictions = predictor.get_predictions(processed_image)
        # Save the processed image
        save_image(BB_IMAGE_PATH, bb_image)
        # Output result as JSON
        result = {
            "entities": predictions,
            "image_path": BB_IMAGE_PATH,
        }
        print(json.dumps(result))  # JSON output for the Node.js server
    else:
        print(json.dumps({"error": "Failed to scan the document."}))

if __name__ == "__main__":
    input_image_path = sys.argv[1]
    #input_image_path = "./static/media/resize_image.jpg"
    main(input_image_path)
