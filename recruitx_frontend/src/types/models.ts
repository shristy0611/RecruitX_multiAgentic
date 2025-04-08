// recruitx_frontend/src/types/models.ts

// Basic structure based on backend Pydantic schemas
// Adjust fields and types as necessary based on actual backend responses

export interface JobAnalysis {
  // Define fields based on schemas.job.JobAnalysis
  required_skills: string[];
  preferred_skills?: string[];
  minimum_experience?: string;
  education?: string;
  responsibilities: string[];
  job_type?: string;
  salary_range?: string;
  company_culture?: string;
  benefits?: string[];
  industry?: string;
  seniority_level?: string;
  market_insights?: {
      skill_demand?: {
          high_demand_skills?: string[];
          trending_skills?: string[];
      };
      salary_insights?: string;
      industry_outlook?: string;
  };
  reasoning?: string;
}

export interface Job {
  id: number;
  title?: string; // Assuming title might be added or inferred
  filename: string;
  description_raw: string;
  analysis?: JobAnalysis | null;
  created_at: string; // Assuming ISO string format
  updated_at?: string | null;
}

export interface CandidateContactInfo {
  name?: string | null;
  email?: string | null;
  phone?: string | null;
  location?: string | null;
}

export interface CandidateWorkExperience {
  company?: string | null;
  title?: string | null;
  dates?: string | null;
  responsibilities?: string[];
}

export interface CandidateEducation {
  institution?: string | null;
  degree?: string | null;
  field_of_study?: string | null;
  graduation_date?: string | null;
}

export interface CandidateProject {
    name?: string | null;
    description?: string | null;
    technologies?: string[];
}

export interface CandidateAnalysis {
  // Define fields based on schemas.candidate.CandidateAnalysis
  candidate_id: number;
  contact_info?: CandidateContactInfo | null;
  summary?: string | null;
  skills?: string[];
  work_experience?: CandidateWorkExperience[];
  education?: CandidateEducation[];
  certifications?: string[];
  projects?: CandidateProject[];
  languages?: string[];
  overall_profile?: string | null;
}

export interface Candidate {
  id: number;
  name?: string; // May come from analysis
  filename: string;
  resume_raw: string;
  analysis?: CandidateAnalysis | null;
  created_at: string; // Assuming ISO string format
  updated_at?: string | null;
}

export interface Score {
  id: number;
  job_id: number;
  candidate_id: number;
  score: number; // Assuming overall_score becomes score
  explanation?: string | null;
  // Include candidate details if the backend endpoint provides them
  candidate?: Candidate;
  created_at: string;
} 