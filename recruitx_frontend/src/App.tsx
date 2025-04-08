import React from 'react'
import { Routes, Route, Link } from 'react-router-dom'
import JDUploadPage from './pages/JDUploadPage'
import CVUploadPage from './pages/CVUploadPage'
import JobListPage from './pages/JobListPage'
import CandidateListPage from './pages/CandidateListPage'
import JobDetailPage from './pages/JobDetailPage'
import CandidateDetailPage from './pages/CandidateDetailPage'
import ComparisonPage from './pages/ComparisonPage'
import './App.css'

// Simple component for the home page
const HomePage: React.FC = () => (
  <div className="text-center">
    <h1 className="text-3xl font-bold mb-4">Welcome to RecruitX</h1>
    <p className="mb-6">Your AI-powered recruitment assistant.</p>
    
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto mb-8">
      {/* Upload Section */}
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-semibold mb-3">Upload Documents</h2>
        <div className="space-y-3">
          <Link
            to="/upload-jd"
            className="block w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-lg"
          >
            Upload Job Description
          </Link>
          <Link
            to="/upload-cv"
            className="block w-full bg-green-600 hover:bg-green-700 text-white py-2 px-4 rounded-lg"
          >
            Upload Candidate CV
          </Link>
        </div>
      </div>
      
      {/* Browse Section */}
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-semibold mb-3">Browse Database</h2>
        <div className="space-y-3">
          <Link
            to="/jobs"
            className="block w-full bg-purple-600 hover:bg-purple-700 text-white py-2 px-4 rounded-lg"
          >
            View All Jobs
          </Link>
          <Link
            to="/candidates"
            className="block w-full bg-orange-600 hover:bg-orange-700 text-white py-2 px-4 rounded-lg"
          >
            View All Candidates
          </Link>
        </div>
      </div>
    </div>
    
    {/* Feature Highlights */}
    <div className="max-w-4xl mx-auto">
      <h2 className="text-xl font-semibold mb-3">Key Features</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-md">
          <h3 className="font-medium mb-2">AI-Powered Analysis</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Automatically extract key information from job descriptions and resumes.
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-md">
          <h3 className="font-medium mb-2">Intelligent Matching</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Score candidates against job requirements with our Agentic RAG system.
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-md">
          <h3 className="font-medium mb-2">Side-by-Side Comparison</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Compare multiple candidates in a detailed view to make better hiring decisions.
          </p>
        </div>
      </div>
    </div>
  </div>
)

function App() {
  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100 flex flex-col items-center justify-center p-4">
      {/* We might add a persistent header/nav bar outside the Routes later */}
      <div className="w-full max-w-7xl">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/upload-jd" element={<JDUploadPage />} />
          <Route path="/upload-cv" element={<CVUploadPage />} />
          <Route path="/jobs" element={<JobListPage />} />
          <Route path="/jobs/:jobId" element={<JobDetailPage />} />
          <Route path="/candidates" element={<CandidateListPage />} />
          <Route path="/candidates/:candidateId" element={<CandidateDetailPage />} />
          <Route path="/comparison/:jobId" element={<ComparisonPage />} />
          {/* Add other routes here as pages are created */}
          {/* e.g., <Route path="/jobs/:jobId" element={<JobDetailPage />} /> */}
          {/* e.g., <Route path="/candidates/:candidateId" element={<CandidateDetailPage />} /> */}
        </Routes>
      </div>
    </div>
  )
}

export default App
