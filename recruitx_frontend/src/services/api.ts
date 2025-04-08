// recruitx_frontend/src/services/api.ts

// Import types
import { Job, Candidate, Score, JobAnalysis, CandidateAnalysis } from '../types/models';

// Read the base URL from environment variables (Vite specific)
// Make sure to define VITE_API_BASE_URL in your .env file (e.g., VITE_API_BASE_URL=http://localhost:8000)
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1'; // Default relative path if not set

/**
 * Represents a successful API response.
 */
interface ApiSuccessResponse<T> {
  success: true;
  data: T;
  status: number;
}

/**
 * Represents a failed API response.
 */
interface ApiErrorResponse {
  success: false;
  error: string; // Error message
  status?: number; // HTTP status code, if available
  details?: any; // Optional additional error details from the server
}

/**
 * Standardized type for API call results.
 */
type ApiResponse<T> = ApiSuccessResponse<T> | ApiErrorResponse;

/**
 * Handles the common logic for making a fetch request and processing the response.
 * @param endpoint The API endpoint (relative to API_BASE_URL).
 * @param options The fetch options (method, headers, body, etc.).
 * @returns Promise<ApiResponse<T>>
 */
async function handleFetch<T>(endpoint: string, options: RequestInit): Promise<ApiResponse<T>> {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const response = await fetch(url, options);

    if (!response.ok) {
      let errorDetails: any = null;
      try {
        errorDetails = await response.json();
      } catch (e) {
        try {
             errorDetails = await response.text();
        } catch (textErr) {
             errorDetails = "Could not parse error response.";
        }
      }
      const errorMessage = `HTTP error! Status: ${response.status}${typeof errorDetails === 'string' && errorDetails ? ': ' + errorDetails : ''}`;
      console.error('API Error:', errorMessage, 'Details:', errorDetails);
      return {
        success: false,
        error: errorMessage,
        status: response.status,
        details: typeof errorDetails !== 'string' ? errorDetails : undefined,
      };
    }

    // Handle potential empty body for 204 No Content or similar
    if (response.status === 204) {
        return { success: true, data: null as T, status: response.status };
    }

    const data: T = await response.json();
    return { success: true, data, status: response.status };

  } catch (error) {
    console.error('Network or other fetch error:', error);
    const errorMessage = error instanceof Error ? error.message : 'An unknown network error occurred';
    return { success: false, error: errorMessage };
  }
}

// --- Upload Functions ---

/**
 * Uploads a Job Description file.
 * @param file The File object to upload.
 * @returns Promise<ApiResponse<Job>> - The response data type depends on your backend endpoint.
 */
export async function uploadJobDescription(file: File): Promise<ApiResponse<Job>> {
  const formData = new FormData();
  formData.append('file', file);
  // Add required form fields
  formData.append('title', file.name.split('.')[0]); // Use filename as title
  formData.append('company', 'Default Company'); // Add default company
  formData.append('location', 'Remote'); // Add default location
  return handleFetch<Job>('/jobs/upload', { method: 'POST', body: formData });
}

/**
 * Uploads a Candidate CV file.
 * @param file The File object to upload.
 * @returns Promise<ApiResponse<Candidate>> - The response data type depends on your backend endpoint.
 */
export async function uploadCandidateCV(file: File): Promise<ApiResponse<Candidate>> {
  const formData = new FormData();
  formData.append('file', file);
  // Add required form fields
  formData.append('name', file.name.split('.')[0]); // Use filename as candidate name
  formData.append('email', 'candidate@example.com'); // Add default email
  formData.append('phone', '+1234567890'); // Add default phone
  return handleFetch<Candidate>('/candidates/upload', { method: 'POST', body: formData });
}

// --- GET Functions ---

/**
 * Fetches a list of jobs.
 */
export async function getJobList(skip: number = 0, limit: number = 100): Promise<ApiResponse<Job[]>> {
  return handleFetch<Job[]>(`/jobs/?skip=${skip}&limit=${limit}`, {
    method: 'GET',
  });
}

/**
 * Fetches details for a specific job.
 */
export async function getJobDetails(jobId: number): Promise<ApiResponse<Job>> {
  return handleFetch<Job>(`/jobs/${jobId}`, {
    method: 'GET',
  });
}

/**
 * Fetches a list of candidates.
 */
export async function getCandidateList(skip: number = 0, limit: number = 100): Promise<ApiResponse<Candidate[]>> {
  return handleFetch<Candidate[]>(`/candidates/?skip=${skip}&limit=${limit}`, {
    method: 'GET',
  });
}

/**
 * Fetches details for a specific candidate.
 */
export async function getCandidateDetails(candidateId: number): Promise<ApiResponse<Candidate>> {
  return handleFetch<Candidate>(`/candidates/${candidateId}`, {
    method: 'GET',
  });
}

/**
 * Fetches scores for candidates related to a specific job.
 */
export async function getScoresForJob(jobId: number): Promise<ApiResponse<Score[]>> {
  return handleFetch<Score[]>(`/scores/job/${jobId}`, {
    method: 'GET',
  });
}

/**
 * Fetches scores for a specific candidate across all jobs.
 */
export async function getScoresForCandidate(candidateId: number): Promise<ApiResponse<Score[]>> {
  return handleFetch<Score[]>(`/scores/candidate/${candidateId}`, {
    method: 'GET',
  });
}

/**
 * Fetches a specific score by its ID.
 */
export async function getScoreDetails(scoreId: number): Promise<ApiResponse<Score>> {
  return handleFetch<Score>(`/scores/${scoreId}`, {
    method: 'GET',
  });
}

// --- Trigger Analysis/Scoring Functions (POST/PUT) ---

/**
 * Triggers the analysis process for a specific job.
 */
export async function triggerJobAnalysis(jobId: number): Promise<ApiResponse<JobAnalysis>> { // Assuming analysis result is returned
    return handleFetch<JobAnalysis>(`/jobs/${jobId}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }, // No body needed, just trigger
    });
}

/**
 * Triggers the analysis process for a specific candidate.
 */
export async function triggerCandidateAnalysis(candidateId: number): Promise<ApiResponse<CandidateAnalysis>> { // Assuming analysis result is returned
    return handleFetch<CandidateAnalysis>(`/candidates/${candidateId}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
    });
}

/**
 * Triggers the scoring process for a specific candidate against a specific job.
 */
export async function triggerScoreCalculation(jobId: number, candidateId: number): Promise<ApiResponse<Score>> { // Assuming the new score is returned
    return handleFetch<Score>(`/scores/job/${jobId}/candidate/${candidateId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
    });
}

// Example for a GET request (add as needed later)
// export async function getJobList(): Promise<ApiResponse<Job[]>> {
//   return handleFetch<Job[]>('/jobs/', {
//     method: 'GET',
//   });
// } 