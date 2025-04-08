import React, { useState, useCallback } from 'react';

interface FileUploadProps {
  onFileSelect: (file: File | null) => void;
  acceptedFileTypes?: string; // e.g., '.pdf,.docx,.txt'
  label?: string;
}

const FileUpload: React.FC<FileUploadProps> = ({
  onFileSelect,
  acceptedFileTypes = '.pdf,.docx,.txt', // Default accepted types
  label = 'Upload File',
}) => {
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);

  const handleFileChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0] || null;
      setSelectedFileName(file ? file.name : null);
      onFileSelect(file);
    },
    [onFileSelect]
  );

  const inputId = `file-upload-${label.replace(/\s+/g, '-').toLowerCase()}`;

  return (
    <div className="w-full max-w-md mx-auto my-4">
      <label
        htmlFor={inputId}
        className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300"
      >
        {label}
      </label>
      <div className="flex items-center justify-center w-full">
        <label
          htmlFor={inputId}
          className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 dark:hover:bg-bray-800 dark:bg-gray-700 hover:bg-gray-100 dark:border-gray-600 dark:hover:border-gray-500 dark:hover:bg-gray-600 transition duration-150 ease-in-out"
        >
          <div className="flex flex-col items-center justify-center pt-5 pb-6">
            <svg
              className="w-8 h-8 mb-4 text-gray-500 dark:text-gray-400"
              aria-hidden="true"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 20 16"
            >
              <path
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"
              />
            </svg>
            <p className="mb-2 text-sm text-gray-500 dark:text-gray-400">
              <span className="font-semibold">Click to upload</span> or drag and drop
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              {acceptedFileTypes.replace(/,/g, ', ')}
            </p>
          </div>
          <input
            id={inputId}
            type="file"
            className="hidden"
            accept={acceptedFileTypes}
            onChange={handleFileChange}
          />
        </label>
      </div>
      {selectedFileName && (
        <p className="mt-2 text-sm text-center text-gray-600 dark:text-gray-400">
          Selected file: <span className="font-medium">{selectedFileName}</span>
        </p>
      )}
    </div>
  );
};

export default FileUpload; 