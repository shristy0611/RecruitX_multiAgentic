import React, { useState } from 'react';
import FileUpload from '../components/FileUpload';
import { uploadCandidateCV } from '../services/api'; // Import the API service function

const CVUploadPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<any | null>(null);

  const handleFileSelect = (file: File | null) => {
    setSelectedFile(file);
    setUploadStatus(null);
    setUploadResult(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadStatus('Please select a file first.');
      return;
    }

    setIsUploading(true);
    setUploadStatus(`Uploading ${selectedFile.name}...`);
    setUploadResult(null);

    const response = await uploadCandidateCV(selectedFile);

    setIsUploading(false);

    if (response.success) {
      setUploadStatus(`Successfully uploaded ${selectedFile.name}!`);
      setUploadResult(response.data);
      console.log('Upload successful:', response.data);
      // Optionally clear the file: setSelectedFile(null);
    } else {
      let errorMessage = `Upload failed: ${response.error}`;
      if (response.details && response.details.detail) {
         errorMessage += ` - ${response.details.detail}`;
      }
      setUploadStatus(errorMessage);
      console.error('Upload failed:', response);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-semibold mb-6 text-gray-800 dark:text-white text-center">
        Upload Candidate CV / Resume
      </h1>

      <FileUpload
        onFileSelect={handleFileSelect}
        acceptedFileTypes=".pdf,.docx,.txt"
        label="Select Candidate CV File"
      />

      <div className="text-center mt-6">
        <button
          onClick={handleUpload}
          disabled={!selectedFile || isUploading}
          className="px-6 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition duration-150 ease-in-out"
        >
          {isUploading ? 'Uploading...' : 'Upload CV'}
        </button>
      </div>

      {uploadStatus && (
        <p className={`mt-4 text-center text-sm ${uploadStatus.toLowerCase().includes('failed') || uploadStatus.toLowerCase().includes('error') ? 'text-red-600' : 'text-green-600'}`}>
          {uploadStatus}
        </p>
      )}

      {/* Optional: Display info from the successful upload result */}
      {uploadResult && (
         <div className="mt-4 p-3 bg-green-100 border border-green-300 rounded-md text-center text-sm text-green-800">
             <p>Upload successful! Received data:</p>
             <pre className="mt-2 text-left text-xs bg-white p-2 rounded overflow-x-auto">{JSON.stringify(uploadResult, null, 2)}</pre>
         </div>
      )}
    </div>
  );
};

export default CVUploadPage; 