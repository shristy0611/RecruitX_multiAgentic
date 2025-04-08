import React, { useState } from 'react';
import FileUpload from '../components/FileUpload';
import { uploadJobDescription } from '../services/api'; // Import the API service function

const JDUploadPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  // Optional: Store the successful response data (e.g., the created Job ID)
  const [uploadResult, setUploadResult] = useState<any | null>(null);
  const [debugInfo, setDebugInfo] = useState<string | null>(null);

  const handleFileSelect = (file: File | null) => {
    setSelectedFile(file);
    setUploadStatus(null);
    setUploadResult(null);
    setDebugInfo(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadStatus('Please select a file first.');
      return;
    }

    setIsUploading(true);
    setUploadStatus(`Uploading ${selectedFile.name}...`);
    setUploadResult(null);
    setDebugInfo(null);

    try {
      const response = await uploadJobDescription(selectedFile);
      
      setIsUploading(false);

      // Debug info
      setDebugInfo(JSON.stringify({
        fileName: selectedFile.name,
        fileSize: selectedFile.size,
        fileType: selectedFile.type,
        responseStatus: response.status,
        responseSuccess: response.success,
        timestamp: new Date().toISOString()
      }, null, 2));

      if (response.success) {
        setUploadStatus(`Successfully uploaded ${selectedFile.name}!`);
        setUploadResult(response.data); // Store response data if needed
        console.log('Upload successful:', response.data);
        // Optionally clear the file: setSelectedFile(null);
      } else {
        // Detailed error message combining API error and potential details
        let errorMessage = `Upload failed: ${response.error}`;
        if (response.details && response.details.detail) {
          // Check if backend provided specific 'detail' message (common in FastAPI)
          errorMessage += ` - ${response.details.detail}`;
        }
        setUploadStatus(errorMessage);
        console.error('Upload failed:', response);
      }
    } catch (error) {
      setIsUploading(false);
      setUploadStatus(`Upload error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      console.error('Upload exception:', error);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-semibold mb-6 text-gray-800 dark:text-white text-center">
        Upload Job Description
      </h1>

      <FileUpload
        onFileSelect={handleFileSelect}
        acceptedFileTypes=".pdf,.docx,.txt"
        label="Select Job Description File"
      />

      <div className="text-center mt-6">
        <button
          onClick={handleUpload}
          disabled={!selectedFile || isUploading}
          className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition duration-150 ease-in-out"
        >
          {isUploading ? 'Uploading...' : 'Upload JD'}
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

      {/* Debug information */}
      {debugInfo && (
        <div className="mt-4 p-3 bg-gray-100 border border-gray-300 rounded-md text-center text-sm text-gray-800">
          <p>Debug Info:</p>
          <pre className="mt-2 text-left text-xs bg-white p-2 rounded overflow-x-auto">{debugInfo}</pre>
        </div>
      )}
    </div>
  );
};

export default JDUploadPage; 