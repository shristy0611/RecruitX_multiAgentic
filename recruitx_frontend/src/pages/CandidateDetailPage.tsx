import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getCandidateDetails } from '../services/api';
import { Candidate, CandidateAnalysis, CandidateWorkExperience, CandidateEducation, CandidateProject, CandidateContactInfo } from '../types/models';

// Helper component to display analysis sections nicely (copied from JobDetailPage, maybe move to components later)
const AnalysisSection: React.FC<{ title: string; data: any; className?: string }> = ({ title, data, className }) => {
  if (data === null || data === undefined || (Array.isArray(data) && data.length === 0) || (typeof data === 'object' && !Array.isArray(data) && Object.keys(data).length === 0 && !(data instanceof Date))) {
    return null; // Don't render empty/null sections
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
  } else if (typeof data === 'object' && data !== null) {
    // Render specific object types differently if needed, otherwise default to JSON
     if (title === "Contact Information") {
      const contact = data as CandidateContactInfo;
      content = (
        <ul className="text-sm">
          {contact.name && <li><strong>Name:</strong> {contact.name}</li>}
          {contact.email && <li><strong>Email:</strong> {contact.email}</li>}
          {contact.phone && <li><strong>Phone:</strong> {contact.phone}</li>}
          {contact.location && <li><strong>Location:</strong> {contact.location}</li>}
        </ul>
      )
    } else if (title === "Work Experience") {
        const experiences = data as CandidateWorkExperience[];
        content = (
          <ul className="space-y-3">
            {experiences.map((exp, index) => (
              <li key={index} className="border-l-2 border-gray-300 dark:border-gray-600 pl-3">
                <p className="font-medium">{exp.title} at {exp.company}</p>
                {exp.dates && <p className="text-xs text-gray-500 dark:text-gray-400">{exp.dates}</p>}
                {exp.responsibilities && exp.responsibilities.length > 0 && (
                  <ul className="list-disc list-inside pl-4 mt-1 text-xs">
                    {exp.responsibilities.map((resp, r_idx) => <li key={r_idx}>{resp}</li>)}
                  </ul>
                )}
              </li>
            ))}
          </ul>
        );
    } else if (title === "Education") {
        const educations = data as CandidateEducation[];
        content = (
            <ul className="space-y-2">
            {educations.map((edu, index) => (
              <li key={index}>
                <p className="font-medium">{edu.institution}</p>
                {edu.degree && <p className="text-sm">{edu.degree}{edu.field_of_study ? ` in ${edu.field_of_study}` : ''}</p>}
                {edu.graduation_date && <p className="text-xs text-gray-500 dark:text-gray-400">Graduated: {edu.graduation_date}</p>}
              </li>
            ))}
          </ul>
        );
    } else if (title === "Projects") {
        const projects = data as CandidateProject[];
        content = (
            <ul className="space-y-3">
            {projects.map((proj, index) => (
              <li key={index} className="border-l-2 border-gray-300 dark:border-gray-600 pl-3">
                <p className="font-medium">{proj.name || 'Unnamed Project'}</p>
                {proj.description && <p className="text-sm mt-1">{proj.description}</p>}
                {proj.technologies && proj.technologies.length > 0 && (
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Technologies: {proj.technologies.join(', ')}</p>
                )}
              </li>
            ))}
          </ul>
        );
    } else {
        // Default fallback for other objects
        content = <pre className="whitespace-pre-wrap break-words bg-gray-100 dark:bg-gray-700 p-2 rounded text-xs">{JSON.stringify(data, null, 2)}</pre>;
    }
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

const CandidateDetailPage: React.FC = () => {
  const { candidateId } = useParams<{ candidateId: string }>();
  const [candidate, setCandidate] = useState<Candidate | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!candidateId) {
      setError('Candidate ID not found in URL.');
      setIsLoading(false);
      return;
    }

    const fetchCandidateDetails = async () => {
      setIsLoading(true);
      setError(null);
      const response = await getCandidateDetails(Number(candidateId)); 
      if (response.success) {
        setCandidate(response.data);
      } else {
        setError(`Failed to fetch candidate details: ${response.error}`);
      }
      setIsLoading(false);
    };

    fetchCandidateDetails();
  }, [candidateId]);

  if (isLoading) {
    return <div className="text-center p-4">Loading candidate details...</div>;
  }

  if (error) {
    return <div className="text-center p-4 text-red-600">Error: {error}</div>;
  }

  if (!candidate) {
    return <div className="text-center p-4">Candidate not found.</div>;
  }

  const analysis = candidate.analysis as CandidateAnalysis | null;
  const candidateName = analysis?.contact_info?.name || candidate.filename;

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="mb-4">
            <Link to="/candidates" className="text-blue-600 hover:underline">&larr; Back to Candidates List</Link>
        </div>
        <h1 className="text-3xl font-bold mb-2 text-gray-800 dark:text-white">
            Candidate: {candidateName}
        </h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
            ID: {candidate.id} | Uploaded: {new Date(candidate.created_at).toLocaleString()}
        </p>

        <div className="bg-white dark:bg-gray-800 shadow-md rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-700 dark:text-gray-200 border-b pb-2">Raw Resume Text</h2>
            <pre className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap break-words bg-gray-50 dark:bg-gray-700 p-4 rounded max-h-96 overflow-y-auto mb-6">
                {candidate.resume_raw || 'No raw resume text available.'}
            </pre>

            <h2 className="text-xl font-semibold mb-4 text-gray-700 dark:text-gray-200 border-b pb-2">Analysis Results</h2>
            {analysis ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4">
                    <AnalysisSection title="Contact Information" data={analysis.contact_info} />
                    <AnalysisSection title="Summary" data={analysis.summary} />
                    <AnalysisSection title="Skills" data={analysis.skills} className="md:col-span-2" />
                    <AnalysisSection title="Work Experience" data={analysis.work_experience} className="md:col-span-2" />
                    <AnalysisSection title="Education" data={analysis.education} className="md:col-span-2"/>
                    <AnalysisSection title="Certifications" data={analysis.certifications} />
                    <AnalysisSection title="Languages" data={analysis.languages} />
                    <AnalysisSection title="Projects" data={analysis.projects} className="md:col-span-2"/>
                    <AnalysisSection title="Overall Profile Assessment" data={analysis.overall_profile} className="md:col-span-2" />
                 </div>
            ) : (
                <p className="text-gray-600 dark:text-gray-400">Analysis has not been run for this candidate yet.</p>
            )}
        </div>
    </div>
  );
};

export default CandidateDetailPage; 