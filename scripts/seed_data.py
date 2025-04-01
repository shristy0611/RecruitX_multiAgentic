import requests
import json
import logging
import os
import sys
from sqlalchemy.orm import Session

# Adjust path to import from the app
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from recruitx_app.core.database import SessionLocal, engine
from recruitx_app.models.candidate import Candidate
from recruitx_app.models.job import Job

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API endpoint
BASE_URL = "http://localhost:8000/api/v1"
UPLOAD_JOB_URL = f"{BASE_URL}/jobs/upload"

# --- Generated Test Data --- 

# 10 Job Descriptions (Simplified for brevity, real ones would be much longer)
JOBS_DATA = [
    {
        "title": "Senior Python Developer", "company": "Tech Innovations Inc.", "location": "Remote",
        "description_raw": "Seeking a Senior Python Developer with 5+ years of experience in FastAPI, Django, and PostgreSQL. Strong understanding of microservices, cloud platforms (AWS/GCP), and CI/CD pipelines required. Experience with machine learning libraries is a plus."
    },
    {
        "title": "Junior Data Scientist", "company": "Data Insights Co.", "location": "New York, NY",
        "description_raw": "Entry-level Data Scientist position. Requires a Master's degree in CS, Statistics, or related field. Must know Python, R, SQL, and common ML algorithms (regression, classification, clustering). Familiarity with data visualization tools (Tableau, PowerBI) expected."
    },
    {
        "title": "Cloud Solutions Architect", "company": "Global Cloud Services", "location": "London, UK",
        "description_raw": "Experienced Cloud Architect needed to design and implement scalable AWS solutions. 7+ years experience with core AWS services (EC2, S3, RDS, Lambda), infrastructure as code (Terraform/CloudFormation), and containerization (Docker, Kubernetes). AWS certification preferred."
    },
    {
        "title": "UX/UI Designer", "company": "Creative Apps Ltd.", "location": "San Francisco, CA",
        "description_raw": "Talented UX/UI Designer to create intuitive interfaces for mobile and web applications. Portfolio demonstrating proficiency in Figma/Sketch, wireframing, prototyping, and user research required. 3+ years experience in a product design role."
    },
    {
        "title": "Project Manager - Agile", "company": "Agile Solutions Group", "location": "Austin, TX",
        "description_raw": "Certified Scrum Master (CSM) or PMP needed to manage software development projects. 5+ years experience leading agile teams, managing backlogs, facilitating ceremonies, and using project management tools like Jira or Asana."
    },
    {
        "title": "Machine Learning Engineer", "company": "AI Core Labs", "location": "Berlin, Germany",
        "description_raw": "ML Engineer to build, train, and deploy machine learning models. Strong background in Python, TensorFlow/PyTorch, MLOps practices, and experience with large datasets. PhD or Master's in a quantitative field required."
    },
    {
        "title": "Frontend Developer (React)", "company": "Web Wizards Inc.", "location": "Remote",
        "description_raw": "Mid-level Frontend Developer proficient in React, Redux, JavaScript (ES6+), HTML5, CSS3. Experience with TypeScript, Next.js, and consuming REST APIs is essential. 3+ years of professional frontend experience."
    },
    {
        "title": "DevOps Engineer", "company": "Infra Automate", "location": "Seattle, WA",
        "description_raw": "Seeking a DevOps Engineer to build and maintain CI/CD pipelines, manage cloud infrastructure (Azure preferred), and implement monitoring solutions. Experience with scripting (Bash/Python), Docker, Kubernetes, and automation tools (Ansible/Puppet) needed. 4+ years relevant experience."
    },
    {
        "title": "Healthcare Data Analyst", "company": "Health Analytics Partners", "location": "Boston, MA",
        "description_raw": "Data Analyst with experience in the healthcare domain. Must be proficient in SQL, data warehousing concepts, and statistical analysis. Familiarity with HIPAA regulations and healthcare data standards (HL7, FHIR) is a strong plus. Bachelor's degree required."
    },
    {
        "title": "Senior Software Engineer (Java)", "company": "Enterprise Systems Corp.", "location": "Chicago, IL",
        "description_raw": "Experienced Java developer (8+ years) with expertise in Spring Boot, microservices architecture, relational databases (Oracle/Postgres), and messaging queues (Kafka/RabbitMQ). Proven ability to lead design and implementation of complex backend systems."
    }
]

# 10 Candidate Resumes (Simplified)
CANDIDATES_DATA = [
    {
        "name": "Alice Martin", "email": "alice.m@example.com", "phone": "111-222-3333",
        "resume_raw": "Highly skilled Python Developer with 6 years experience. Proficient in FastAPI, PostgreSQL, AWS, Docker. Led development of three major microservices. Contributed to open-source ML projects."
    },
    {
        "name": "Bob Chen", "email": "bob.c@example.com", "phone": "222-333-4444",
        "resume_raw": "Recent Master's graduate in Data Science. Strong foundation in Python, R, SQL, scikit-learn, Keras. Completed thesis on customer churn prediction. Internship experience with Tableau."
    },
    {
        "name": "Charlie Davis", "email": "charlie.d@example.com", "phone": "333-444-5555",
        "resume_raw": "AWS Certified Solutions Architect with 8 years experience designing cloud infrastructure. Expertise in EC2, S3, VPC, Terraform, Kubernetes. Migrated large monolithic applications to AWS microservices."
    },
    {
        "name": "Diana Evans", "email": "diana.e@example.com", "phone": "444-555-6666",
        "resume_raw": "Creative UX/UI Designer with 4 years experience. Proficient in Figma, user research, and interaction design. Portfolio includes mobile apps and complex web platforms. Passionate about accessibility."
    },
    {
        "name": "Ethan Foster", "email": "ethan.f@example.com", "phone": "555-666-7777",
        "resume_raw": "Agile Project Manager (CSM) with 6 years leading software teams. Skilled in Jira, sprint planning, risk management, and stakeholder communication. Successfully delivered 10+ projects on time."
    },
    {
        "name": "Fiona Garcia", "email": "fiona.g@example.com", "phone": "666-777-8888",
        "resume_raw": "Machine Learning Engineer with 5 years experience. Expertise in Python, PyTorch, computer vision, and NLP. Developed and deployed models for image recognition and sentiment analysis. Published research papers."
    },
    {
        "name": "George Harris", "email": "george.h@example.com", "phone": "777-888-9999",
        "resume_raw": "Frontend Developer with 4 years experience specializing in React and TypeScript. Built responsive UIs using Next.js and Material UI. Strong understanding of state management and API integration."
    },
    {
        "name": "Hannah Ivanova", "email": "hannah.i@example.com", "phone": "888-999-0000",
        "resume_raw": "DevOps Engineer with 5 years experience in Azure, CI/CD using Azure DevOps, Docker, Kubernetes, and Ansible. Automated infrastructure deployment and improved monitoring systems significantly."
    },
    {
        "name": "Ian Jenkins", "email": "ian.j@example.com", "phone": "999-000-1111",
        "resume_raw": "Data Analyst with 3 years experience in healthcare analytics. Proficient in SQL, Python (Pandas), and building dashboards in PowerBI. Familiar with EMR data and HIPAA compliance."
    },
    {
        "name": "Julia King", "email": "julia.k@example.com", "phone": "000-111-2222",
        "resume_raw": "Senior Java Developer with 10 years experience. Expert in Spring Boot, microservices, Kafka, and Oracle. Led backend team for a large e-commerce platform. Strong system design skills."
    }
]

# --- Seeding Logic --- 

def clear_data(db: Session):
    logger.info("Clearing existing jobs and candidates...")
    db.query(Candidate).delete()
    db.query(Job).delete()
    db.commit()
    logger.info("Data cleared.")

def seed_jobs():
    logger.info("Seeding jobs...")
    job_ids = []
    for job_data in JOBS_DATA:
        try:
            # We need to simulate a file upload
            # Create a dummy file in memory
            files = {'file': ('job_description.txt', job_data["description_raw"], 'text/plain')}
            payload = {
                'title': job_data["title"],
                'company': job_data.get("company", ""),
                'location': job_data.get("location", "")
            }
            response = requests.post(UPLOAD_JOB_URL, data=payload, files=files)
            response.raise_for_status()  # Raise an exception for bad status codes
            created_job = response.json()
            job_ids.append(created_job['id'])
            logger.info(f"Created job: {created_job['title']} (ID: {created_job['id']})")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to upload job '{job_data['title']}': {e}")
            if e.response is not None:
                logger.error(f"Response body: {e.response.text}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while seeding job '{job_data['title']}': {e}")
    logger.info("Job seeding complete.")
    return job_ids

def seed_candidates(db: Session):
    logger.info("Seeding candidates...")
    candidate_ids = []
    for candidate_data in CANDIDATES_DATA:
        try:
            db_candidate = Candidate(**candidate_data)
            db.add(db_candidate)
            db.commit()
            db.refresh(db_candidate)
            candidate_ids.append(db_candidate.id)
            logger.info(f"Created candidate: {db_candidate.name} (ID: {db_candidate.id})")
        except Exception as e:
            logger.error(f"Failed to create candidate '{candidate_data['name']}': {e}")
            db.rollback()
    logger.info("Candidate seeding complete.")
    return candidate_ids

if __name__ == "__main__":
    logger.info("Starting data seeding process...")
    db = SessionLocal()
    try:
        # Clear existing data first
        # clear_data(db)

        # Seed new data
        job_ids = seed_jobs()
        candidate_ids = seed_candidates(db)

        logger.info("--- Seeding Summary ---")
        logger.info(f"Created {len(job_ids)} jobs with IDs: {job_ids}")
        logger.info(f"Created {len(candidate_ids)} candidates with IDs: {candidate_ids}")
        logger.info("Seeding process completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during the seeding process: {e}")
    finally:
        db.close() 