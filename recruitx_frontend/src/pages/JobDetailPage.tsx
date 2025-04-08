import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getJobDetails } from '../services/api';
import { Job, JobAnalysis } from '../types/models';

// Helper component to display analysis sections nicely
const AnalysisSection: React.FC<{ title: string; data: any; className?: string }> = ({ title, data, className }) => {
  if (!data || (Array.isArray(data) && data.length === 0)) {
    return null; // Don't render empty sections
  }

  let content;
  if (typeof data === 'string' || typeof data === 'number') {
    content = <p>{data}</p>;
  } else if (Array.isArray(data)) {
    content = (
      <ul className="list-disc list-inside pl-4">
        {data.map((item, index) => (
          <li key={index}>{typeof item === 'object' ? JSON.stringify(item) : item}</li>
        ))}
      </ul>
    );
  } else if (typeof data === 'object') {
    // Render nested objects (like market insights)
    content = <pre className="whitespace-pre-wrap break-words bg-gray-100 dark:bg-gray-700 p-2 rounded text-xs">{JSON.stringify(data, null, 2)}</pre>;
  } else {
    content = <p>Invalid data format</p>;
  }

  return (
    <div className={`mb-4 ${className}`}>
      <h3 className="text-md font-semibold text-gray-700 dark:text-gray-300 mb-1">{title}</h3>
      <div className="text-sm text-gray-600 dark:text-gray-400">{content}</div>
    </div>
  );
};


const JobDetailPage: React.FC = () => {
  const { jobId } = useParams<{ jobId: string }>();
  const [job, setJob] = useState<Job | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) {
      setError('Job ID not found in URL.');
      setIsLoading(false);
      return;
    }

    const fetchJobDetails = async () => {
      setIsLoading(true);
      setError(null);
      const response = await getJobDetails(Number(jobId)); // Convert string ID to number
      if (response.success) {
        setJob(response.data);
      } else {
        setError(`Failed to fetch job details: ${response.error}`);
      }
      setIsLoading(false);
    };

    fetchJobDetails();
  }, [jobId]); // Re-fetch if jobId changes

  if (isLoading) {
    return <div className="text-center p-4">Loading job details...</div>;
  }

  if (error) {
    return <div className="text-center p-4 text-red-600">Error: {error}</div>;
  }

  if (!job) {
    return <div className="text-center p-4">Job not found.</div>;
  }

  const analysis = job.analysis as JobAnalysis | null; // Type assertion for easier access

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <div className="mb-4 flex justify-between items-center">
        <Link to="/jobs" className="text-blue-600 hover:underline">&larr; Back to Jobs List</Link>
        <Link 
          to={`/comparison/${job.id}`} 
          className="bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-lg flex items-center"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
            <path d="M5 3a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2V5a2 2 0 00-2-2H5zM5 11a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2v-2a2 2 0 00-2-2H5zM11 5a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V5zM11 13a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
          </svg>
          Compare Candidates
        </Link>
      </div>
      <h1 className="text-3xl font-bold mb-2 text-gray-800 dark:text-white">
        Job Details: {job.title || job.filename}
      </h1>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
         ID: {job.id} | Uploaded: {new Date(job.created_at).toLocaleString()}
      </p>

      <div className="bg-white dark:bg-gray-800 shadow-md rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-700 dark:text-gray-200 border-b pb-2">Raw Description</h2>
        <pre className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap break-words bg-gray-50 dark:bg-gray-700 p-4 rounded max-h-96 overflow-y-auto mb-6">
           {job.description_raw || 'No raw description available.'}
        </pre>

        <h2 className="text-xl font-semibold mb-4 text-gray-700 dark:text-gray-200 border-b pb-2">Analysis Results</h2>
        {job.analysis ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4">
                <AnalysisSection title="Required Skills" data={analysis?.required_skills} />
                <AnalysisSection title="Preferred Skills" data={analysis?.preferred_skills} />
                <AnalysisSection title="Minimum Experience" data={analysis?.minimum_experience} />
                <AnalysisSection title="Education" data={analysis?.education} />
                <AnalysisSection title="Job Type" data={analysis?.job_type} />
                <AnalysisSection title="Seniority Level" data={analysis?.seniority_level} />
                <AnalysisSection title="Industry" data={analysis?.industry} />
                <AnalysisSection title="Salary Range" data={analysis?.salary_range} />
                <AnalysisSection title="Benefits" data={analysis?.benefits} />
                <AnalysisSection title="Company Culture" data={analysis?.company_culture} className="md:col-span-2"/>
                <AnalysisSection title="Responsibilities" data={analysis?.responsibilities} className="md:col-span-2" />
                <AnalysisSection title="Market Insights" data={analysis?.market_insights} className="md:col-span-2" />
                <AnalysisSection title="Analysis Reasoning" data={analysis?.reasoning} className="md:col-span-2" />
            </div>
        ) : (
          <p className="text-gray-600 dark:text-gray-400">Analysis has not been run for this job yet.</p>
        )}
      </div>
    </div>
  );
};

export default JobDetailPage; 