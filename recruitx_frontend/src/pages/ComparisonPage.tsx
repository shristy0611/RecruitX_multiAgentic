import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { Job, Candidate, Score } from '../types/models';
import { getJobDetails, getScoresForJob, getCandidateDetails } from '../services/api';

interface ComparisonPageParams {
  jobId?: string;
}

const ComparisonPage: React.FC = () => {
  const { jobId } = useParams<keyof ComparisonPageParams>() as ComparisonPageParams;
  const navigate = useNavigate();
  
  const [job, setJob] = useState<Job | null>(null);
  const [scores, setScores] = useState<Score[]>([]);
  const [candidates, setCandidates] = useState<Map<number, Candidate>>(new Map());
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCandidates, setSelectedCandidates] = useState<number[]>([]);
  const [maxScoreDisplay, setMaxScoreDisplay] = useState<number>(5);
  const [activeTab, setActiveTab] = useState<'side-by-side' | 'comparison-table'>('side-by-side');

  // Fetch job details and scores
  useEffect(() => {
    const fetchJobAndScores = async () => {
      if (!jobId) {
        setError('No job ID provided');
        setLoading(false);
        return;
      }

      try {
        const jobResponse = await getJobDetails(parseInt(jobId));
        if (!jobResponse.success) {
          throw new Error(jobResponse.error);
        }

        const scoresResponse = await getScoresForJob(parseInt(jobId));
        if (!scoresResponse.success) {
          throw new Error(scoresResponse.error);
        }

        setJob(jobResponse.data);
        
        // Sort scores by score value descending
        const sortedScores = [...scoresResponse.data].sort((a, b) => b.score - a.score);
        setScores(sortedScores);
        
        // Select top 3 candidates by default
        if (sortedScores.length > 0) {
          setSelectedCandidates(sortedScores.slice(0, 3).map(score => score.candidate_id));
        }

        // Fetch candidate details for each score
        const candidatesMap = new Map<number, Candidate>();
        
        for (const score of sortedScores) {
          const candidateResponse = await getCandidateDetails(score.candidate_id);
          if (candidateResponse.success) {
            candidatesMap.set(score.candidate_id, candidateResponse.data);
          }
        }
        
        setCandidates(candidatesMap);
        setLoading(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An unknown error occurred');
        setLoading(false);
      }
    };

    fetchJobAndScores();
  }, [jobId]);

  // Toggle a candidate from the comparison
  const toggleCandidate = (candidateId: number) => {
    if (selectedCandidates.includes(candidateId)) {
      setSelectedCandidates(selectedCandidates.filter(id => id !== candidateId));
    } else {
      setSelectedCandidates([...selectedCandidates, candidateId]);
    }
  };

  // Calculate the CSS width for columns based on number of selected candidates
  const getColumnWidth = () => {
    const selectedCount = selectedCandidates.length;
    if (selectedCount === 0) return '100%';
    return `${100 / (selectedCount + 1)}%`; // +1 for the job column
  };

  // Helper function to determine match level between job requirement and candidate qualification
  const getMatchLevel = (requirement: string, candidateQualifications: string[] | undefined): 'match' | 'partial' | 'none' => {
    if (!candidateQualifications || candidateQualifications.length === 0) return 'none';
    
    const reqLower = requirement.toLowerCase();
    
    // Check for direct match
    if (candidateQualifications.some(qual => qual.toLowerCase() === reqLower)) {
      return 'match';
    }
    
    // Check for partial match (one is substring of the other)
    if (candidateQualifications.some(qual => 
      qual.toLowerCase().includes(reqLower) || reqLower.includes(qual.toLowerCase())
    )) {
      return 'partial';
    }
    
    return 'none';
  };

  // Helper function to render match status with appropriate color
  const renderMatchStatus = (matchLevel: 'match' | 'partial' | 'none') => {
    switch (matchLevel) {
      case 'match':
        return <span className="text-green-500 font-medium">Strong Match</span>;
      case 'partial':
        return <span className="text-yellow-500 font-medium">Partial Match</span>;
      case 'none':
        return <span className="text-red-500 font-medium">No Match</span>;
    }
  };

  if (loading) {
    return (
      <div className="p-8 text-center">
        <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-blue-500 mx-auto"></div>
        <p className="mt-4">Loading comparison data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 text-center text-red-500">
        <h2 className="text-2xl font-bold mb-4">Error</h2>
        <p>{error}</p>
        <button 
          onClick={() => navigate(-1)} 
          className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Go Back
        </button>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="p-8 text-center">
        <h2 className="text-2xl font-bold mb-4">Job Not Found</h2>
        <button 
          onClick={() => navigate('/jobs')} 
          className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Back to Jobs
        </button>
      </div>
    );
  }

  return (
    <div className="p-4">
      <div className="mb-6 flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold mb-2">
            Candidate Comparison
          </h1>
          <h2 className="text-xl">
            for {job.title || 'Untitled Job'} (ID: {job.id})
          </h2>
        </div>
        <div>
          <Link 
            to={`/jobs/${job.id}`} 
            className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600 mr-2"
          >
            Back to Job
          </Link>
          <Link 
            to="/jobs" 
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            All Jobs
          </Link>
        </div>
      </div>

      {/* Candidate Selection */}
      <div className="mb-6 p-4 bg-white dark:bg-gray-800 rounded shadow">
        <h3 className="text-lg font-semibold mb-3">Select Candidates to Compare (Top Scores)</h3>
        
        <div className="flex flex-wrap gap-2 mb-4">
          {scores.slice(0, maxScoreDisplay).map(score => (
            <button
              key={score.candidate_id}
              onClick={() => toggleCandidate(score.candidate_id)}
              className={`px-3 py-1 rounded text-sm font-medium ${
                selectedCandidates.includes(score.candidate_id)
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
              }`}
            >
              {candidates.get(score.candidate_id)?.name || `Candidate ${score.candidate_id}`} ({score.score}/100)
            </button>
          ))}
        </div>
        
        {scores.length > maxScoreDisplay && (
          <button 
            onClick={() => setMaxScoreDisplay(prev => prev + 5)}
            className="text-blue-500 hover:underline text-sm"
          >
            Show more candidates...
          </button>
        )}
      </div>

      {/* View Toggle */}
      <div className="mb-6 flex border-b border-gray-200">
        <button
          className={`py-2 px-4 font-medium ${
            activeTab === 'side-by-side' 
              ? 'border-b-2 border-blue-500 text-blue-500' 
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('side-by-side')}
        >
          Side-by-Side Requirements
        </button>
        <button
          className={`py-2 px-4 font-medium ${
            activeTab === 'comparison-table' 
              ? 'border-b-2 border-blue-500 text-blue-500' 
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('comparison-table')}
        >
          Comparison Table
        </button>
      </div>

      {selectedCandidates.length > 0 ? (
        <>
          {/* Side-by-Side Requirements View */}
          {activeTab === 'side-by-side' && (
            <div className="bg-white dark:bg-gray-800 rounded shadow overflow-x-auto mb-6">
              <h3 className="text-xl font-semibold p-4 border-b dark:border-gray-700">
                Job Requirements vs. Candidate Qualifications
              </h3>
              
              {/* Skills Section */}
              <div className="p-4 border-b dark:border-gray-700">
                <h4 className="text-lg font-medium mb-3 text-blue-600">Skills Requirements</h4>
                
                {job.analysis?.required_skills && job.analysis.required_skills.length > 0 ? (
                  <div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <h5 className="font-medium mb-2">Required Skills</h5>
                        <ul className="list-disc list-inside pl-2 space-y-1">
                          {job.analysis?.required_skills.map((skill, index) => (
                            <li key={index} className="text-gray-700 dark:text-gray-300">{skill}</li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <h5 className="font-medium mb-2">Preferred Skills</h5>
                        <ul className="list-disc list-inside pl-2 space-y-1">
                          {job.analysis?.preferred_skills && job.analysis.preferred_skills.length > 0 ? (
                            job.analysis?.preferred_skills.map((skill, index) => (
                              <li key={index} className="text-gray-700 dark:text-gray-300">{skill}</li>
                            ))
                          ) : (
                            <li className="text-gray-500 dark:text-gray-400">No preferred skills specified</li>
                          )}
                        </ul>
                      </div>
                    </div>
                    
                    <div className="mt-6 space-y-6">
                      {selectedCandidates.map(candidateId => {
                        const candidate = candidates.get(candidateId);
                        const score = scores.find(s => s.candidate_id === candidateId);
                        
                        return (
                          <div key={candidateId} className="p-3 bg-gray-50 dark:bg-gray-900 rounded">
                            <h5 className="font-medium mb-2 flex justify-between">
                              <span>{candidate?.name || `Candidate ${candidateId}`}</span>
                              <span className="text-blue-500">Score: {score?.score}/100</span>
                            </h5>
                            
                            <div className="space-y-4">
                              <div>
                                <h6 className="text-sm font-medium mb-1">Required Skills Assessment:</h6>
                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-2">
                                  {job.analysis?.required_skills && job.analysis.required_skills.map((skill, index) => {
                                    const matchLevel = getMatchLevel(skill, candidate?.analysis?.skills);
                                    
                                    return (
                                      <div key={index} className="flex justify-between border-b border-gray-200 dark:border-gray-700 pb-1">
                                        <span>{skill}</span>
                                        <span>{renderMatchStatus(matchLevel)}</span>
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>
                              
                              {job.analysis?.preferred_skills && job.analysis.preferred_skills.length > 0 && (
                                <div>
                                  <h6 className="text-sm font-medium mb-1">Preferred Skills Assessment:</h6>
                                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-2">
                                    {job.analysis?.preferred_skills.map((skill, index) => {
                                      const matchLevel = getMatchLevel(skill, candidate?.analysis?.skills);
                                      
                                      return (
                                        <div key={index} className="flex justify-between border-b border-gray-200 dark:border-gray-700 pb-1">
                                          <span>{skill}</span>
                                          <span>{renderMatchStatus(matchLevel)}</span>
                                        </div>
                                      );
                                    })}
                                  </div>
                                </div>
                              )}
                              
                              <div>
                                <h6 className="text-sm font-medium mb-1">All Candidate Skills:</h6>
                                <div className="flex flex-wrap gap-2">
                                  {candidate?.analysis?.skills && candidate.analysis.skills.length > 0 ? (
                                    candidate.analysis.skills.map((skill, index) => (
                                      <span 
                                        key={index} 
                                        className={`text-xs px-2 py-1 rounded ${
                                          job.analysis?.required_skills?.some(req => req.toLowerCase() === skill.toLowerCase())
                                            ? 'bg-green-100 dark:bg-green-800 text-green-800 dark:text-green-100' 
                                            : job.analysis?.preferred_skills?.some(pref => pref.toLowerCase() === skill.toLowerCase())
                                              ? 'bg-yellow-100 dark:bg-yellow-800 text-yellow-800 dark:text-yellow-100'
                                              : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                                        }`}
                                      >
                                        {skill}
                                      </span>
                                    ))
                                  ) : (
                                    <span className="text-gray-500 dark:text-gray-400">No skills listed</span>
                                  )}
                                </div>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ) : (
                  <p className="text-gray-500 dark:text-gray-400">No skills requirements specified in the job description.</p>
                )}
              </div>
              
              {/* Experience Section */}
              <div className="p-4 border-b dark:border-gray-700">
                <h4 className="text-lg font-medium mb-3 text-blue-600">Experience Requirements</h4>
                
                <div className="mb-4">
                  <h5 className="font-medium mb-2">Required Experience</h5>
                  <p className="text-gray-700 dark:text-gray-300">
                    {job.analysis?.minimum_experience || "No specific experience requirement mentioned"}
                  </p>
                </div>
                
                <div className="mt-6 space-y-6">
                  {selectedCandidates.map(candidateId => {
                    const candidate = candidates.get(candidateId);
                    
                    return (
                      <div key={candidateId} className="p-3 bg-gray-50 dark:bg-gray-900 rounded">
                        <h5 className="font-medium mb-2">
                          {candidate?.name || `Candidate ${candidateId}`}'s Experience
                        </h5>
                        
                        {candidate?.analysis?.work_experience && candidate.analysis.work_experience.length > 0 ? (
                          <div className="space-y-2">
                            {candidate.analysis.work_experience.map((exp, index) => (
                              <div key={index} className="border-l-4 border-blue-500 pl-3 py-1">
                                <div className="font-medium">{exp.title} at {exp.company}</div>
                                <div className="text-sm text-gray-500 dark:text-gray-400">{exp.dates}</div>
                                {exp.responsibilities && exp.responsibilities.length > 0 && (
                                  <ul className="text-xs list-disc list-inside mt-1">
                                    {exp.responsibilities.map((resp, i) => (
                                      <li key={i}>{resp}</li>
                                    ))}
                                  </ul>
                                )}
                              </div>
                            ))}
                          </div>
                        ) : (
                          <p className="text-gray-500 dark:text-gray-400">No experience listed</p>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
              
              {/* Education Section */}
              <div className="p-4 border-b dark:border-gray-700">
                <h4 className="text-lg font-medium mb-3 text-blue-600">Education Requirements</h4>
                
                <div className="mb-4">
                  <h5 className="font-medium mb-2">Required Education</h5>
                  <p className="text-gray-700 dark:text-gray-300">
                    {job.analysis?.education || "No specific education requirement mentioned"}
                  </p>
                </div>
                
                <div className="mt-6 space-y-6">
                  {selectedCandidates.map(candidateId => {
                    const candidate = candidates.get(candidateId);
                    
                    return (
                      <div key={candidateId} className="p-3 bg-gray-50 dark:bg-gray-900 rounded">
                        <h5 className="font-medium mb-2">
                          {candidate?.name || `Candidate ${candidateId}`}'s Education
                        </h5>
                        
                        {candidate?.analysis?.education && candidate.analysis.education.length > 0 ? (
                          <div className="space-y-2">
                            {candidate.analysis.education.map((edu, index) => (
                              <div key={index}>
                                <div className="font-medium">{edu.degree} in {edu.field_of_study}</div>
                                <div className="text-sm">{edu.institution}</div>
                                <div className="text-xs text-gray-500 dark:text-gray-400">{edu.graduation_date}</div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <p className="text-gray-500 dark:text-gray-400">No education listed</p>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
              
              {/* Certifications Section */}
              <div className="p-4">
                <h4 className="text-lg font-medium mb-3 text-blue-600">Certifications</h4>
                
                <div className="mt-4 space-y-6">
                  {selectedCandidates.map(candidateId => {
                    const candidate = candidates.get(candidateId);
                    
                    return (
                      <div key={candidateId} className="p-3 bg-gray-50 dark:bg-gray-900 rounded">
                        <h5 className="font-medium mb-2">
                          {candidate?.name || `Candidate ${candidateId}`}'s Certifications
                        </h5>
                        
                        {candidate?.analysis?.certifications && candidate.analysis.certifications.length > 0 ? (
                          <ul className="list-disc list-inside pl-2">
                            {candidate.analysis.certifications.map((cert, index) => (
                              <li key={index}>{cert}</li>
                            ))}
                          </ul>
                        ) : (
                          <p className="text-gray-500 dark:text-gray-400">No certifications listed</p>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* Original Comparison Table View */}
          {activeTab === 'comparison-table' && (
            <div className="bg-white dark:bg-gray-800 rounded shadow overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-gray-100 dark:bg-gray-700">
                    <th className="p-3 text-left font-semibold" style={{ width: getColumnWidth() }}>Criteria</th>
                    {selectedCandidates.map(candidateId => {
                      const candidate = candidates.get(candidateId);
                      const score = scores.find(s => s.candidate_id === candidateId);
                      
                      return (
                        <th key={candidateId} className="p-3 text-left font-semibold" style={{ width: getColumnWidth() }}>
                          <div className="flex flex-col">
                            <span>{candidate?.name || `Candidate ${candidateId}`}</span>
                            <span className="text-blue-500 text-sm">Score: {score?.score}/100</span>
                            <Link 
                              to={`/candidates/${candidateId}`}
                              className="text-xs text-blue-500 hover:underline"
                            >
                              View Profile
                            </Link>
                          </div>
                        </th>
                      );
                    })}
                  </tr>
                </thead>
                <tbody>
                  {/* Overall Match */}
                  <tr className="border-b dark:border-gray-700">
                    <td className="p-3 font-medium bg-gray-50 dark:bg-gray-900">Overall Match</td>
                    {selectedCandidates.map(candidateId => {
                      const score = scores.find(s => s.candidate_id === candidateId);
                      return (
                        <td key={candidateId} className="p-3">
                          <div className="w-full bg-gray-200 rounded-full h-4 dark:bg-gray-700">
                            <div 
                              className="bg-blue-600 h-4 rounded-full"
                              style={{ width: `${score?.score || 0}%` }}
                            >
                            </div>
                          </div>
                          <span className="text-sm">{score?.score || 0}%</span>
                        </td>
                      );
                    })}
                  </tr>

                  {/* Skills */}
                  <tr className="border-b dark:border-gray-700">
                    <td className="p-3 font-medium bg-gray-50 dark:bg-gray-900">Skills</td>
                    {selectedCandidates.map(candidateId => {
                      const candidate = candidates.get(candidateId);
                      return (
                        <td key={candidateId} className="p-3">
                          {candidate?.analysis?.skills ? (
                            <ul className="list-disc list-inside text-sm">
                              {candidate.analysis.skills.map((skill, index) => (
                                <li key={index}>{skill}</li>
                              ))}
                            </ul>
                          ) : (
                            <span className="text-gray-500">No skills data</span>
                          )}
                        </td>
                      );
                    })}
                  </tr>

                  {/* Experience */}
                  <tr className="border-b dark:border-gray-700">
                    <td className="p-3 font-medium bg-gray-50 dark:bg-gray-900">Experience</td>
                    {selectedCandidates.map(candidateId => {
                      const candidate = candidates.get(candidateId);
                      return (
                        <td key={candidateId} className="p-3">
                          {candidate?.analysis?.work_experience && candidate.analysis.work_experience.length > 0 ? (
                            <div className="text-sm">
                              {candidate.analysis.work_experience.map((exp, index) => (
                                <div key={index} className="mb-2">
                                  <div className="font-medium">{exp.title} at {exp.company}</div>
                                  <div className="text-gray-500 dark:text-gray-400">{exp.dates}</div>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <span className="text-gray-500">No experience data</span>
                          )}
                        </td>
                      );
                    })}
                  </tr>

                  {/* Education */}
                  <tr className="border-b dark:border-gray-700">
                    <td className="p-3 font-medium bg-gray-50 dark:bg-gray-900">Education</td>
                    {selectedCandidates.map(candidateId => {
                      const candidate = candidates.get(candidateId);
                      return (
                        <td key={candidateId} className="p-3">
                          {candidate?.analysis?.education && candidate.analysis.education.length > 0 ? (
                            <div className="text-sm">
                              {candidate.analysis.education.map((edu, index) => (
                                <div key={index} className="mb-2">
                                  <div className="font-medium">{edu.degree} in {edu.field_of_study}</div>
                                  <div>{edu.institution}</div>
                                  <div className="text-gray-500 dark:text-gray-400">{edu.graduation_date}</div>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <span className="text-gray-500">No education data</span>
                          )}
                        </td>
                      );
                    })}
                  </tr>

                  {/* Certifications */}
                  <tr className="border-b dark:border-gray-700">
                    <td className="p-3 font-medium bg-gray-50 dark:bg-gray-900">Certifications</td>
                    {selectedCandidates.map(candidateId => {
                      const candidate = candidates.get(candidateId);
                      return (
                        <td key={candidateId} className="p-3">
                          {candidate?.analysis?.certifications && candidate.analysis.certifications.length > 0 ? (
                            <ul className="list-disc list-inside text-sm">
                              {candidate.analysis.certifications.map((cert, index) => (
                                <li key={index}>{cert}</li>
                              ))}
                            </ul>
                          ) : (
                            <span className="text-gray-500">No certifications data</span>
                          )}
                        </td>
                      );
                    })}
                  </tr>

                  {/* Explanation */}
                  <tr className="dark:border-gray-700">
                    <td className="p-3 font-medium bg-gray-50 dark:bg-gray-900">Match Explanation</td>
                    {selectedCandidates.map(candidateId => {
                      const score = scores.find(s => s.candidate_id === candidateId);
                      return (
                        <td key={candidateId} className="p-3">
                          <div className="text-sm whitespace-pre-wrap">
                            {score?.explanation || 'No explanation available'}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                </tbody>
              </table>
            </div>
          )}
        </>
      ) : (
        <div className="bg-white dark:bg-gray-800 p-8 text-center rounded shadow">
          <h3 className="text-xl mb-4">Please select candidates to compare</h3>
          <p className="text-gray-500 dark:text-gray-400">
            Use the buttons above to select candidates for side-by-side comparison.
          </p>
        </div>
      )}
    </div>
  );
};

export default ComparisonPage; 