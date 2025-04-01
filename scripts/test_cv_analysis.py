import sys
import os
import asyncio
import json

# Adjust the Python path to include the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from recruitx_app.core.database import get_db
from recruitx_app.services.candidate_service import CandidateService
from recruitx_app.schemas.candidate import CandidateCreate
from sqlalchemy.orm import Session

# Create a test resume
TEST_CV = """
JOHN DEVELOPER
Software Engineer
john.developer@email.com | (555) 123-4567 | San Francisco, CA | linkedin.com/in/johndeveloper

SUMMARY
Experienced software engineer with 6+ years of expertise in full-stack development, specializing in Python, JavaScript, and cloud technologies. Proven track record of delivering scalable applications and optimizing performance.

SKILLS
Programming Languages: Python, JavaScript, TypeScript, Java, SQL
Frameworks & Libraries: Django, Flask, React, Node.js, Express.js
Cloud & DevOps: AWS (EC2, S3, Lambda), Docker, Kubernetes, CI/CD
Databases: PostgreSQL, MongoDB, Redis
Tools: Git, JIRA, Jenkins, Prometheus, Grafana

WORK EXPERIENCE

Senior Software Engineer | TechCorp Inc. | Jan 2022 - Present
- Architected and implemented microservices-based applications using Python, Django, and React
- Reduced application response time by 40% through database optimization and caching strategies
- Led a team of 5 engineers in developing a new customer-facing portal with 10k+ daily users
- Implemented CI/CD pipelines that decreased deployment time by 65%

Software Engineer | DataSystems LLC | June 2019 - Dec 2021
- Developed RESTful APIs using Flask and integrated with third-party services
- Built and maintained data pipelines processing 500GB+ daily using Python and AWS
- Collaborated with cross-functional teams to deliver features on time and within scope
- Implemented automated testing that increased code coverage by 35%

Junior Developer | StartupX | Aug 2017 - May 2019
- Contributed to front-end development using React and TypeScript
- Assisted in database design and optimization
- Participated in agile development processes and daily stand-ups

EDUCATION
Bachelor of Science in Computer Science
University of California, Berkeley
Graduated: May 2017 | GPA: 3.8/4.0

PROJECTS
E-commerce Platform (2021)
- Built a full-stack e-commerce platform using Django, React, and PostgreSQL
- Implemented secure payment processing and user authentication

Data Visualization Tool (2020)
- Created an interactive dashboard for visualizing large datasets
- Utilized D3.js and Python for data processing and visualization

CERTIFICATIONS
- AWS Certified Solutions Architect - Associate (2021)
- MongoDB Certified Developer (2020)

LANGUAGES
- English (Native)
- Spanish (Professional working proficiency)
"""

async def test_cv_analysis():
    """Test CV analysis functionality with a sample resume."""
    print("--- Starting CV Analysis Test ---")
    
    try:
        # Get DB session
        db = next(get_db())
        
        # Create CandidateService instance
        candidate_service = CandidateService()
        
        # Create a test candidate with our sample CV using the Pydantic model
        candidate_data = CandidateCreate(
            name="John Developer",
            email="john.developer@email.com",
            phone="(555) 123-4567",
            resume_raw=TEST_CV
        )
        
        print(f"Creating test candidate with name: {candidate_data.name}")
        candidate = candidate_service.create_candidate(db, candidate_data=candidate_data)
        print(f"Created candidate with ID: {candidate.id}")
        
        # Analyze the CV
        print(f"\nAnalyzing candidate CV...")
        analysis_result = await candidate_service.analyze_cv(db, candidate_id=candidate.id)
        
        if not analysis_result:
            print("CV analysis failed!")
            return
        
        # Print the analysis results formatted
        print("\nCV Analysis Results:")
        print(json.dumps(analysis_result, indent=2))
        
        print("\n--- CV Analysis Test Successful ---")
        
        # Cleanup - remove the test candidate
        db.delete(candidate)
        db.commit()
        print(f"Deleted test candidate (ID: {candidate.id})")
        
    except Exception as e:
        print(f"\n--- CV Analysis Test Failed ---")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_cv_analysis()) 