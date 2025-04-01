import sys
import os
import asyncio
import json

# Adjust the Python path to include the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from recruitx_app.core.database import get_db
from recruitx_app.services.job_service import JobService
from recruitx_app.schemas.job import JobCreate
from sqlalchemy.orm import Session

# Create a test job description
TEST_JD = """
Job Title: Full Stack Developer

About Us:
At TechInnovate, we're building the next generation of cloud-based enterprise solutions. Our team is dedicated to creating scalable, reliable software that solves real business problems.

Responsibilities:
- Design and develop high-quality web applications using React, Node.js, and Python
- Collaborate with cross-functional teams to define, design, and ship new features
- Ensure the technical feasibility of UI/UX designs
- Optimize applications for maximum speed and scalability
- Implement security and data protection measures

Requirements:
- 3+ years of experience in full stack development
- Proficient in React, Node.js, and Python
- Experience with RESTful APIs and microservices architecture
- Knowledge of front-end technologies (HTML5, CSS3, JavaScript)
- Familiarity with database technology (SQL, NoSQL)
- Experience with cloud platforms (AWS, Azure, or GCP)
- Bachelor's degree in Computer Science or related field
- Excellent problem-solving and communication skills

Nice to Have:
- Experience with container orchestration (Kubernetes, Docker)
- Knowledge of CI/CD pipelines
- Experience with Agile development methodologies
- Open source contributions
"""

async def test_jd_analysis():
    """Test JD analysis functionality with a sample job description."""
    print("--- Starting JD Analysis Test ---")
    
    try:
        # Get DB session
        db = next(get_db())
        
        # Create JobService instance
        job_service = JobService()
        
        # Create a test job with our sample JD using the Pydantic model
        job_data = JobCreate(
            title="Full Stack Developer",
            company="TechInnovate",
            location="Remote",
            description_raw=TEST_JD
        )
        
        print(f"Creating test job with title: {job_data.title}")
        job = job_service.create_job(db, job_data=job_data)
        print(f"Created job with ID: {job.id}")
        
        # Analyze the job
        print(f"\nAnalyzing job description...")
        analysis_result = await job_service.analyze_job(db, job_id=job.id)
        
        if not analysis_result:
            print("JD analysis failed!")
            return
        
        # Print the analysis results formatted
        print("\nJD Analysis Results:")
        print(json.dumps(analysis_result, indent=2))
        
        print("\n--- JD Analysis Test Successful ---")
        
        # Cleanup - remove the test job
        db.delete(job)
        db.commit()
        print(f"Deleted test job (ID: {job.id})")
        
    except Exception as e:
        print(f"\n--- JD Analysis Test Failed ---")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_jd_analysis()) 