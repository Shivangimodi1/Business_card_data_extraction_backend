// Import required modules
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const multer = require('multer');
const tesseract = require('tesseract.js');
const { exec } = require('child_process'); // For running Python scripts
const fs = require('fs');
const path = require('path');

// Initialize Express app and configure environment
require('dotenv').config();
const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use('/static', express.static(path.join(__dirname, 'static')));
// Ensure your static file directory is set correctly
app.use(express.static(path.join(__dirname, 'uploads')));
// Configure multer for file uploads
const upload = multer({ dest: 'uploads/' });

// Utility functions

/**
 * Updates the CSV file with extracted data.
 * @param {Object} extractedData - Extracted data from the business card.
 */
const updateCSV = (extractedData) => {
  const csvFilePath = path.join(__dirname, 'uploads', 'business_card_data.csv');
  const headers = 'Name,Business,Designation,Phone,Email Address,Website\n';

  // Extract values from arrays or use an empty string if they are missing
  const newData = [
    extractedData.NAME?.[0] || '',
    extractedData.ORG?.[0] || '',
    extractedData.DES?.[0] || '',
    extractedData.PHONE?.[0] || '',
    extractedData.EMAIL?.[0] || '',
    extractedData.WEB?.[0] || '',
  ];

  const row = newData.join(',') + '\n';
  if (!fs.existsSync(csvFilePath)) {
    fs.writeFileSync(csvFilePath, headers + row);
  } else {
    fs.appendFileSync(csvFilePath, row);
  }
  console.log('CSV Updated Successfully.');
};

/**
 * Calls the Python NER model to predict data and return bounding boxes.
 * @param {string} imagePath - Path to the input image.
 * @returns {Promise<Object>} - An object containing the image with bounding boxes and extracted entities.
 */
const predictWithNER = (imagePath) => {
  return new Promise((resolve, reject) => {
    exec(`python py_scripts/predictions.py "${imagePath}"`, (error, stdout, stderr) => {
      if (error) {
        console.error('Python execution error:', stderr);
        return reject('NER model execution failed.');
      }

      // Attempt to isolate the JSON part from the stdout
      const jsonStart = stdout.indexOf('{"extracted_data"');
      if (jsonStart !== -1) {
        const jsonString = stdout.substring(jsonStart); // Extract the JSON part
        try {
          const result = JSON.parse(jsonString); // Parse the JSON data
          resolve(result);
        } catch (parseError) {
          console.error('Error parsing NER model output:', parseError);
          reject('Failed to parse NER response.');
        }
      } else {
        console.error('No JSON data found in the output');
        reject('No valid JSON found in the NER response.');
      }
    });
  });
};

// Routes

/**
 * POST route for uploading and processing a business card.
 */
app.post('/upload-card', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: 'No file uploaded.' });
    }
    console.log('Uploaded file:', req.file);

    // Step 1: Call the Python script to process the uploaded image
    const inputImagePath = path.resolve(req.file.path);
    const pythonScript = path.resolve('py_scripts/card_scanner_ner.py');
    const command = `python ${pythonScript} ${inputImagePath}`;

    exec(command, (error, stdout, stderr) => {
      if (error) {
        console.error('Error executing Python script:', error);
        return res.status(500).json({ message: 'Failed to process the card.', error: error.message });
      }

      if (stderr) {
        console.error('Python script error:', stderr);
        return res.status(500).json({ message: 'Error in Python script execution.', error: stderr });
      }
      
      console.log('Python script output:', stdout);

      // Step 2: Parse Python script output (assuming it's JSON formatted)
      let nerResult;
      try {
        // Replace backslashes with forward slashes to avoid JSON parsing issues
        const sanitizedOutput = stdout.replace(/\\/g, '/');
        // Attempt to parse the sanitized output
        nerResult = JSON.parse(sanitizedOutput); 
      } catch (parseError) {
        console.error('Failed to parse Python script output:', parseError);
        return res.status(500).json({ message: 'Invalid Python script output.', error: parseError.message });
      }

      // Step 3: Update CSV with extracted data
      updateCSV(nerResult.entities); // Call updateCSV to save the extracted data into CSV

      // Step 4: Respond with the extracted data and file path
      const responseData = {
        message: 'Card uploaded and processed successfully.',
        extractedData: nerResult.entities, // Adjust based on Python script's output format
        outputImagePath: nerResult.image_path, // Adjust based on Python script's output
      };

      res.status(200).json(responseData);

    });
  } catch (error) {
    console.error('Error processing card:', error);
    res.status(500).json({ message: 'Error processing the card.', error: error.message });
  }
});

/**
 * GET route for downloading the latest CSV file.
 */
app.get('/download-csv', (req, res) => {
  const csvFilePath = path.join(__dirname, 'uploads', 'business_card_data.csv');
  if (fs.existsSync(csvFilePath)) {
    res.download(csvFilePath, 'business_card_data.csv', (err) => {
      if (err) {
        console.error('Error sending file:', err);
        res.status(500).send('Error downloading the file');
      }
    });
  } else {
    res.status(404).send('CSV file not found.');
  }
});

/**
 * GET route for viewing the latest CSV file in the browser.
 */
app.get('/view-csv', (req, res) => {
  const csvFilePath = path.join(__dirname, 'uploads', 'business_card_data.csv');
  res.setHeader('Content-Type', 'text/csv');
  fs.createReadStream(csvFilePath).pipe(res);
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
