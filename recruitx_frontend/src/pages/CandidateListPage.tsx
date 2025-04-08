import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { getCandidateList, triggerCandidateAnalysis } from '../services/api';
import { Candidate } from '../types/models';

const CandidateListPage: React.FC = () => {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [analysisStatus, setAnalysisStatus] = useState<Record<number, string>>({}); // Track analysis status per candidate

  useEffect(() => {
    const fetchCandidates = async () => {
      setIsLoading(true);
      setError(null);
      const response = await getCandidateList();
      if (response.success) {
        setCandidates(response.data);
      } else {
        setError(`Failed to fetch candidates: ${response.error}`);
      }
      setIsLoading(false);
    };

    fetchCandidates();
  }, []); // Run once on mount

  const handleAnalyzeClick = async (candidateId: number) => {
    setAnalysisStatus(prev => ({ ...prev, [candidateId]: 'Analyzing...' }));
    const response = await triggerCandidateAnalysis(candidateId);
    if (response.success) {
        setAnalysisStatus(prev => ({ ...prev, [candidateId]: 'Analysis Complete!' }));
        // Optionally, re-fetch the candidate list or update the specific item
    } else {
        setAnalysisStatus(prev => ({ ...prev, [candidateId]: `Error: ${response.error}` }));
    }
  };

  // Helper function to get candidate name (falls back to filename)
  const getCandidateName = (candidate: Candidate): string => {
    return candidate.analysis?.contact_info?.name || candidate.filename;
  }

  if (isLoading) {
    return <div className="text-center p-4">Loading candidates...</div>;
  }

  if (error) {
    return <div className="text-center p-4 text-red-600">Error: {error}</div>;
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-semibold mb-6 text-gray-800 dark:text-white text-center">
        Candidates
      </h1>

      {candidates.length === 0 ? (
        <p className="text-center text-gray-600 dark:text-gray-400">No candidates uploaded yet.</p>
      ) : (
        <div className="overflow-x-auto shadow-md sm:rounded-lg">
          <table className="w-full text-sm text-left text-gray-500 dark:text-gray-400">
            <thead className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
              <tr>
                <th scope="col" className="px-6 py-3">ID</th>
                <th scope="col" className="px-6 py-3">Name / Filename</th>
                <th scope="col" className="px-6 py-3">Created At</th>
                <th scope="col" className="px-6 py-3">Analysis Status</th>
                <th scope="col" className="px-6 py-3">Actions</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map((candidate) => (
                <tr key={candidate.id} className="bg-white border-b dark:bg-gray-800 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600">
                  <td className="px-6 py-4 font-medium text-gray-900 whitespace-nowrap dark:text-white">
                    {candidate.id}
                  </td>
                  <td className="px-6 py-4">{getCandidateName(candidate)}</td>
                  <td className="px-6 py-4">{new Date(candidate.created_at).toLocaleString()}</td>
                  <td className="px-6 py-4">
                    {candidate.analysis ? (
                       <span className="text-green-600">Analyzed</span>
                    ) : analysisStatus[candidate.id] ? (
                       <span className={analysisStatus[candidate.id].includes('Error') ? 'text-red-600' : 'text-yellow-600'}>
                           {analysisStatus[candidate.id]}
                       </span>
                    ) : (
                       <span className="text-gray-500">Not Analyzed</span>
                    )}
                  </td>
                  <td className="px-6 py-4 space-x-2">
                    <Link 
                      to={`/candidates/${candidate.id}`} // Placeholder for future detail page route
                      className="font-medium text-blue-600 dark:text-blue-500 hover:underline"
                    >
                      View
                    </Link>
                    {!candidate.analysis && !analysisStatus[candidate.id]?.includes('Analyzing') && (
                        <button
                           onClick={() => handleAnalyzeClick(candidate.id)}
                           className="font-medium text-indigo-600 dark:text-indigo-500 hover:underline disabled:opacity-50 disabled:cursor-not-allowed"
                           disabled={analysisStatus[candidate.id]?.includes('Analyzing')}
                        >
                           {analysisStatus[candidate.id]?.includes('Analyzing') ? 'Processing...' : 'Analyze'}
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

export default CandidateListPage; 