import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { getJobList, triggerJobAnalysis } from '../services/api';
import { Job } from '../types/models';

const JobListPage: React.FC = () => {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [analysisStatus, setAnalysisStatus] = useState<Record<number, string>>({}); // Track analysis status per job

  useEffect(() => {
    const fetchJobs = async () => {
      setIsLoading(true);
      setError(null);
      const response = await getJobList();
      if (response.success) {
        setJobs(response.data);
      } else {
        setError(`Failed to fetch jobs: ${response.error}`);
      }
      setIsLoading(false);
    };

    fetchJobs();
  }, []); // Empty dependency array means this runs once on mount

  const handleAnalyzeClick = async (jobId: number) => {
    setAnalysisStatus(prev => ({ ...prev, [jobId]: 'Analyzing...' }));
    const response = await triggerJobAnalysis(jobId);
    if (response.success) {
        setAnalysisStatus(prev => ({ ...prev, [jobId]: 'Analysis Complete!' }));
        // Optionally, re-fetch the job list or update the specific job item
        // to show that analysis is done (e.g., show analysis results link)
    } else {
        setAnalysisStatus(prev => ({ ...prev, [jobId]: `Error: ${response.error}` }));
    }
  };

  if (isLoading) {
    return <div className="text-center p-4">Loading jobs...</div>;
  }

  if (error) {
    return <div className="text-center p-4 text-red-600">Error: {error}</div>;
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-semibold mb-6 text-gray-800 dark:text-white text-center">
        Job Descriptions
      </h1>

      {jobs.length === 0 ? (
        <p className="text-center text-gray-600 dark:text-gray-400">No job descriptions uploaded yet.</p>
      ) : (
        <div className="overflow-x-auto shadow-md sm:rounded-lg">
          <table className="w-full text-sm text-left text-gray-500 dark:text-gray-400">
            <thead className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
              <tr>
                <th scope="col" className="px-6 py-3">Job ID</th>
                <th scope="col" className="px-6 py-3">Filename</th>
                <th scope="col" className="px-6 py-3">Created At</th>
                <th scope="col" className="px-6 py-3">Analysis Status</th>
                <th scope="col" className="px-6 py-3">Actions</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => (
                <tr key={job.id} className="bg-white border-b dark:bg-gray-800 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600">
                  <td className="px-6 py-4 font-medium text-gray-900 whitespace-nowrap dark:text-white">
                    {job.id}
                  </td>
                  <td className="px-6 py-4">{job.filename}</td>
                  <td className="px-6 py-4">{new Date(job.created_at).toLocaleString()}</td>
                  <td className="px-6 py-4">
                    {job.analysis ? (
                       <span className="text-green-600">Analyzed</span>
                    ) : analysisStatus[job.id] ? (
                       <span className={analysisStatus[job.id].includes('Error') ? 'text-red-600' : 'text-yellow-600'}>
                           {analysisStatus[job.id]}
                       </span>
                    ) : (
                       <span className="text-gray-500">Not Analyzed</span>
                    )}
                  </td>
                  <td className="px-6 py-4 space-x-2">
                    <Link 
                      to={`/jobs/${job.id}`} 
                      className="font-medium text-blue-600 dark:text-blue-500 hover:underline"
                    >
                      View
                    </Link>
                    {!job.analysis && !analysisStatus[job.id]?.includes('Analyzing') && (
                        <button
                           onClick={() => handleAnalyzeClick(job.id)}
                           className="font-medium text-indigo-600 dark:text-indigo-500 hover:underline disabled:opacity-50 disabled:cursor-not-allowed"
                           disabled={analysisStatus[job.id]?.includes('Analyzing')}
                        >
                           {analysisStatus[job.id]?.includes('Analyzing') ? 'Processing...' : 'Analyze'}
                        </button>
                    )}
                     {/* Add link/button for viewing scores later */} 
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default JobListPage; 