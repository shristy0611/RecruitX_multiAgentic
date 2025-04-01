import sys
import os
import asyncio
import json

# Adjust the Python path to include the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from recruitx_app.core.database import get_db
from recruitx_app.services.job_service import JobService
from recruitx_app.services.candidate_service import CandidateService
from recruitx_app.services.scoring_service import ScoringService
from recruitx_app.schemas.job import JobCreate
from recruitx_app.schemas.candidate import CandidateCreate
from sqlalchemy.orm import Session

# Sample JD and CV for testing
TEST_JD = """
Job Title: Senior Python Developer

About Us:
We are a leading fintech company developing innovative solutions for the financial industry.

Responsibilities:
- Design, build and maintain efficient, reusable, and reliable Python code
- Implement data storage solutions using PostgreSQL, MongoDB, and Redis
- Develop and integrate RESTful APIs
- Collaborate with frontend developers to integrate user-facing elements
- Perform code reviews and mentor junior developers

Requirements:
- 5+ years of experience in Python development
- Proficient with Python web frameworks (Django or Flask)
- Strong knowledge of ORM libraries and SQL/NoSQL databases
- Experience with Docker, Kubernetes, and CI/CD pipelines
- Understanding of the software development lifecycle
- Bachelor's degree in Computer Science or related field
"""

TEST_CV = """
JOHN DEVELOPER
Software Engineer
john.developer@email.com | (555) 123-4567 | San Francisco, CA

SUMMARY
Experienced software engineer with 6+ years of expertise in Python development, specializing in web applications and cloud deployment. Strong background in developing scalable microservices and RESTful APIs.

SKILLS
Programming Languages: Python, JavaScript, SQL
Frameworks: Django, Flask, React
Databases: PostgreSQL, MongoDB, Redis
DevOps: Docker, Kubernetes, Jenkins, AWS

WORK EXPERIENCE

Senior Python Developer | FinanceTech Inc. | Jan 2021 - Present
- Developed microservices architecture using Django and Flask
- Implemented database solutions with PostgreSQL and MongoDB
- Created RESTful APIs for frontend integration
- Containerized applications using Docker and Kubernetes
- Mentored junior developers and performed code reviews

Software Engineer | WebSolutions LLC | June 2018 - Dec 2020
- Built web applications using Python and Django
- Designed and implemented database schemas
- Developed and maintained RESTful APIs
- Implemented automated testing and CI/CD pipelines

EDUCATION
Bachelor of Science in Computer Science
Stanford University - 2018

CERTIFICATIONS
- AWS Certified Developer - Associate
- MongoDB Certified Developer
"""

async def test_scoring():
    """Test scoring functionality for job-candidate matching."""
    print("--- Starting Scoring Test ---")
    
    try:
        # Get DB session
        db = next(get_db())
        
        # Create service instances
        job_service = JobService()
        candidate_service = CandidateService()
        scoring_service = ScoringService()
        
        # Create a test job using the JobCreate schema
        job_data = JobCreate(
            title="Senior Python Developer",
            company="FinTech Inc.",
            location="San Francisco, CA",
            description_raw=TEST_JD
        )
        
        print(f"Creating test job with title: {job_data.title}")
        job = job_service.create_job(db, job_data=job_data)
        print(f"Created job with ID: {job.id}")
        
        # Create a test candidate using the CandidateCreate schema
        candidate_data = CandidateCreate(
            name="John Developer",
            email="john.developer@email.com",
            phone="(555) 123-4567",
            resume_raw=TEST_CV
        )
        
        print(f"\nCreating test candidate with name: {candidate_data.name}")
        candidate = candidate_service.create_candidate(db, candidate_data=candidate_data)
        print(f"Created candidate with ID: {candidate.id}")
        
        # Analyze the job
        print(f"\nAnalyzing job description...")
        job_analysis = await job_service.analyze_job(db, job_id=job.id)
        
        if not job_analysis:
            print("Job analysis failed!")
            return
            
        print(f"Job analysis successful")
        
        # Analyze the CV
        print(f"\nAnalyzing candidate CV...")
        cv_analysis = await candidate_service.analyze_cv(db, candidate_id=candidate.id)
        
        if not cv_analysis:
            print("CV analysis failed!")
            return
            
        print(f"CV analysis successful")
        
        # Generate the score
        print(f"\nGenerating match score...")
        score = await scoring_service.generate_score(db, job_id=job.id, candidate_id=candidate.id)
        
        if not score:
            print("Score generation failed!")
            return
        
        # Print the score results
        print(f"\nScoring Results:")
        print(f"Overall Score: {score.overall_score}")
        print(f"Explanation: {score.explanation}")
        
        if score.details:
            print("\nDetailed Scoring:")
            print(json.dumps(score.details, indent=2))
        
        print("\n--- Scoring Test Successful ---")
        
        # Cleanup - remove test data
        db.delete(score)
        db.delete(candidate)
        db.delete(job)
        db.commit()
        print(f"Deleted test data")
        
    except Exception as e:
        print(f"\n--- Scoring Test Failed ---")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_scoring()) 