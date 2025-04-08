import aiohttp
import logging
import json
import asyncio
from typing import Dict, Any, Optional, List, Union, TypedDict
from functools import lru_cache
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class APIResponse(TypedDict):
    """Type definition for API responses"""
    success: bool
    data: Optional[Dict[str, Any]]
    error: Optional[str]

class ExternalToolService:
    """
    Service for integrating with external APIs and tools to enhance 
    the Agentic RAG system with additional data points.
    
    This service manages:
    1. Salary benchmarking data
    2. Job market insights
    3. Industry trend information
    """
    
    def __init__(self):
        # Base URLs for external APIs
        self.salary_api_base_url = os.getenv("SALARY_API_URL", "https://api.salarydata.example.com/v1")
        self.market_api_base_url = os.getenv("MARKET_API_URL", "https://api.marketdata.example.com/v1")
        
        # API keys for external services
        self.salary_api_key = os.getenv("SALARY_API_KEY")
        self.market_api_key = os.getenv("MARKET_API_KEY")
        
        # Cache configuration
        self.cache_ttl = int(os.getenv("EXTERNAL_API_CACHE_TTL", "86400"))  # Default: 24 hours
        
        # Default timeout for API requests in seconds
        self.request_timeout = int(os.getenv("EXTERNAL_API_TIMEOUT", "10"))
        
        # Initialize in-memory cache
        self._cache = {}
        self._cache_timestamps = {}
    
    async def _make_api_request(
        self, 
        url: str, 
        method: str = "GET", 
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        use_cache: bool = True,
        cache_key: Optional[str] = None
    ) -> APIResponse:
        """
        Makes an HTTP request to an external API with error handling and caching.
        
        Args:
            url: The URL to request
            method: HTTP method (GET, POST, etc.)
            headers: HTTP headers
            params: URL parameters
            data: Request body for POST/PUT requests
            api_key: API key to use (if not included in headers)
            use_cache: Whether to use cache for this request
            cache_key: Custom cache key (defaults to url+params)
            
        Returns:
            APIResponse with success flag, data or error message
        """
        # Check cache if applicable
        if use_cache and method.upper() == "GET":
            if not cache_key and params:
                # Create a cache key from the URL and params
                param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
                cache_key = f"{url}?{param_str}"
            elif not cache_key:
                cache_key = url
                
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_response
        
        # Prepare headers
        if headers is None:
            headers = {}
            
        # Add API key if provided and not in headers
        if api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {api_key}"
            
        # Default content type for POST/PUT requests with data
        if data and method.upper() in ["POST", "PUT"] and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=data if data and method.upper() in ["POST", "PUT"] else None,
                    timeout=self.request_timeout
                ) as response:
                    response_text = await response.text()
                    
                    try:
                        response_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON response from {url}: {response_text[:200]}")
                        return {
                            "success": False,
                            "data": None,
                            "error": f"Invalid JSON response received. Status: {response.status}"
                        }
                    
                    if response.status >= 200 and response.status < 300:
                        result = {
                            "success": True,
                            "data": response_data,
                            "error": None
                        }
                        
                        # Cache successful GET responses if requested
                        if use_cache and method.upper() == "GET" and cache_key:
                            self._store_in_cache(cache_key, result)
                            
                        return result
                    else:
                        error_msg = response_data.get("error", {}).get("message", "Unknown error")
                        logger.warning(f"API request to {url} failed with status {response.status}: {error_msg}")
                        return {
                            "success": False,
                            "data": None,
                            "error": f"API request failed with status {response.status}: {error_msg}"
                        }
                        
        except aiohttp.ClientError as e:
            logger.error(f"Connection error when accessing {url}: {str(e)}")
            return {
                "success": False,
                "data": None,
                "error": f"Connection error: {str(e)}"
            }
        except asyncio.TimeoutError:
            logger.error(f"Timeout when accessing {url}")
            return {
                "success": False,
                "data": None,
                "error": "Request timed out"
            }
        except Exception as e:
            logger.error(f"Unexpected error when accessing {url}: {str(e)}", exc_info=True)
            return {
                "success": False,
                "data": None,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def _get_from_cache(self, key: str) -> Optional[APIResponse]:
        """Retrieves a value from the cache if it exists and is not expired"""
        if key in self._cache and key in self._cache_timestamps:
            timestamp = self._cache_timestamps[key]
            now = datetime.now()
            
            # Check if the cached value is still valid
            if now - timestamp < timedelta(seconds=self.cache_ttl):
                return self._cache[key]
            else:
                # Clear expired cache entries
                del self._cache[key]
                del self._cache_timestamps[key]
                
        return None
    
    def _store_in_cache(self, key: str, value: APIResponse) -> None:
        """Stores a value in the cache with the current timestamp"""
        self._cache[key] = value
        self._cache_timestamps[key] = datetime.now()
    
    async def get_salary_benchmark(
        self, 
        job_title: str, 
        location: Optional[str] = None,
        experience_years: Optional[int] = None,
        skills: Optional[List[str]] = None
    ) -> APIResponse:
        """
        Retrieves salary benchmark data for a specific job title.
        
        Args:
            job_title: The job title to get salary data for
            location: Optional location to get regional salary data
            experience_years: Optional years of experience
            skills: Optional list of skills to refine the search
            
        Returns:
            APIResponse containing salary benchmark data or error
        """
        # Build request parameters
        params = {
            "title": job_title
        }
        
        if location:
            params["location"] = location
            
        if experience_years is not None:
            params["experience"] = experience_years
            
        if skills:
            params["skills"] = ",".join(skills)
        
        # Check if we have a valid API key and URL
        if self.salary_api_key and not self.salary_api_base_url.endswith("example.com"):
            logger.info(f"Making actual API call to salary service for job title: {job_title}")
            return await self._make_api_request(
                f"{self.salary_api_base_url}/salary/benchmark",
                params=params,
                api_key=self.salary_api_key
            )
        else:
            logger.info(f"Using simulated salary data for job title: {job_title} (no API key or using example domain)")
            # Fall back to simulation if no API key or using the example domain
            simulated_response = self._simulate_salary_data(job_title, location, experience_years, skills)
            return {
                "success": True,
                "data": simulated_response,
                "error": None
            }
    
    async def get_job_market_insights(
        self,
        job_title: str,
        skills: Optional[List[str]] = None,
        location: Optional[str] = None,
        time_period: str = "6months"
    ) -> APIResponse:
        """
        Retrieves job market insights including demand trends.
        
        Args:
            job_title: The job title to get market data for
            skills: Optional list of skills to analyze
            location: Optional location to get regional data
            time_period: Time period for trend analysis (e.g., '6months', '1year')
            
        Returns:
            APIResponse containing market insight data or error
        """
        # Build request parameters
        params = {
            "title": job_title,
            "period": time_period
        }
        
        if location:
            params["location"] = location
            
        if skills:
            params["skills"] = ",".join(skills)
        
        # Check if we have a valid API key and URL
        if self.market_api_key and not self.market_api_base_url.endswith("example.com"):
            logger.info(f"Making actual API call to market data service for job title: {job_title}")
            return await self._make_api_request(
                f"{self.market_api_base_url}/job/market-insights",
                params=params,
                api_key=self.market_api_key
            )
        else:
            logger.info(f"Using simulated market data for job title: {job_title} (no API key or using example domain)")
            # Fall back to simulation if no API key or using the example domain
            simulated_response = self._simulate_market_data(job_title, skills, location, time_period)
            return {
                "success": True,
                "data": simulated_response,
                "error": None
            }
    
    async def get_skill_demand_trends(
        self,
        skills: List[str],
        location: Optional[str] = None,
        time_period: str = "6months"
    ) -> APIResponse:
        """
        Retrieves demand trends for specific skills.
        
        Args:
            skills: List of skills to analyze
            location: Optional location to get regional data
            time_period: Time period for trend analysis (e.g., '6months', '1year')
            
        Returns:
            APIResponse containing skill trend data or error
        """
        if not skills:
            return {
                "success": False,
                "data": None,
                "error": "No skills provided for trend analysis"
            }
        
        # Build request parameters
        params = {
            "skills": ",".join(skills),
            "period": time_period
        }
        
        if location:
            params["location"] = location
        
        # Check if we have a valid API key and URL
        if self.market_api_key and not self.market_api_base_url.endswith("example.com"):
            logger.info(f"Making actual API call to market data service for skill trends: {', '.join(skills[:3])}...")
            return await self._make_api_request(
                f"{self.market_api_base_url}/skills/trends",
                params=params,
                api_key=self.market_api_key
            )
        else:
            logger.info(f"Using simulated skill trend data for skills: {', '.join(skills[:3])}... (no API key or using example domain)")
            # Fall back to simulation if no API key or using the example domain
            simulated_response = self._simulate_skill_trends(skills, location, time_period)
            return {
                "success": True,
                "data": simulated_response,
                "error": None
            }
    
    def _simulate_salary_data(
        self, 
        job_title: str, 
        location: Optional[str], 
        experience_years: Optional[int],
        skills: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Generates realistic simulated salary data"""
        # Base salary ranges for common roles
        base_ranges = {
            "software engineer": {"min": 75000, "max": 150000, "median": 110000},
            "data scientist": {"min": 85000, "max": 160000, "median": 120000},
            "product manager": {"min": 90000, "max": 180000, "median": 135000},
            "ux designer": {"min": 70000, "max": 130000, "median": 95000},
            "marketing manager": {"min": 65000, "max": 120000, "median": 90000},
            "sales representative": {"min": 60000, "max": 110000, "median": 85000},
            "financial analyst": {"min": 70000, "max": 130000, "median": 95000},
            "human resources": {"min": 55000, "max": 105000, "median": 75000},
        }
        
        # Default range if job title not in our sample data
        default_range = {"min": 60000, "max": 120000, "median": 85000}
        
        # Get the base range for this job title
        job_title_lower = job_title.lower()
        salary_range = default_range
        
        for key, value in base_ranges.items():
            if key in job_title_lower:
                salary_range = value
                break
        
        # Location adjustment factors
        location_factors = {
            "san francisco": 1.5,
            "new york": 1.4,
            "boston": 1.3,
            "seattle": 1.25,
            "austin": 1.1,
            "chicago": 1.15,
            "los angeles": 1.3,
            "denver": 1.05,
            "atlanta": 1.0,
            "dallas": 1.05,
            "miami": 1.0,
            "portland": 1.1,
            "remote": 1.0,
        }
        
        # Default location factor
        location_factor = 1.0
        
        # Apply location adjustment if provided
        if location:
            location_lower = location.lower()
            for key, factor in location_factors.items():
                if key in location_lower:
                    location_factor = factor
                    break
        
        # Experience adjustment
        experience_factor = 1.0
        if experience_years is not None:
            if experience_years < 2:
                experience_factor = 0.8
            elif experience_years < 5:
                experience_factor = 1.0
            elif experience_years < 10:
                experience_factor = 1.2
            else:
                experience_factor = 1.4
        
        # Skills adjustment
        skills_factor = 1.0
        premium_skills = ["machine learning", "ai", "blockchain", "react", "kubernetes", "cloud", "aws", "azure", 
                           "devops", "security", "data engineering", "scala", "rust", "golang", "nlp"]
        
        if skills:
            # Calculate premium based on valuable skills
            skill_matches = sum(1 for skill in skills if any(premium in skill.lower() for premium in premium_skills))
            skills_factor = 1.0 + (skill_matches * 0.03)  # 3% bump per premium skill
        
        # Calculate adjusted salary range
        combined_factor = location_factor * experience_factor * skills_factor
        
        return {
            "job_title": job_title,
            "salary_data": {
                "min": int(salary_range["min"] * combined_factor),
                "max": int(salary_range["max"] * combined_factor),
                "median": int(salary_range["median"] * combined_factor),
                "currency": "USD",
                "period": "annual"
            },
            "market_factors": {
                "location_factor": round(location_factor, 2),
                "experience_factor": round(experience_factor, 2),
                "skills_factor": round(skills_factor, 2)
            },
            "reference_date": datetime.now().strftime("%Y-%m-%d"),
            "metadata": {
                "source": "RecruitX Simulated Salary Data",
                "location": location or "United States (Average)",
                "experience_years": experience_years or "All experience levels",
                "skills_considered": skills or []
            }
        }
    
    def _simulate_market_data(
        self,
        job_title: str,
        skills: Optional[List[str]],
        location: Optional[str],
        time_period: str
    ) -> Dict[str, Any]:
        """Generates realistic simulated job market data"""
        # Job growth trends (positive or negative for different roles)
        growth_trends = {
            "software engineer": 0.15,
            "data scientist": 0.22,
            "product manager": 0.12,
            "ux designer": 0.18,
            "marketing manager": 0.05,
            "sales representative": 0.02,
            "financial analyst": 0.08,
            "human resources": 0.04,
        }
        
        # Default growth if job title not in our sample data
        default_growth = 0.08
        
        # Get the growth trend for this job title
        job_title_lower = job_title.lower()
        growth_rate = default_growth
        
        for key, value in growth_trends.items():
            if key in job_title_lower:
                growth_rate = value
                break
        
        # Generate sample data points for the demand trend
        trend_points = []
        current_time = datetime.now()
        
        # Determine number of data points based on time period
        if time_period == "1month":
            num_points = 4  # Weekly
        elif time_period == "3months":
            num_points = 12  # Weekly
        elif time_period == "6months":
            num_points = 6  # Monthly
        else:  # 1year
            num_points = 12  # Monthly
            
        # Generate trend data with slight random variations
        import random
        base_demand = 100  # Starting index
        
        for i in range(num_points):
            # Calculate date for this point
            if time_period in ["1month", "3months"]:
                point_date = (current_time - timedelta(days=(num_points - i) * 7)).strftime("%Y-%m-%d")
            else:
                point_date = (current_time - timedelta(days=(num_points - i) * 30)).strftime("%Y-%m-%d")
                
            # Calculate demand with some randomness for realism
            point_growth = growth_rate * (i / num_points)  # Progressive growth
            variation = random.uniform(-0.05, 0.05)  # +/- 5% random variation
            point_demand = base_demand * (1 + point_growth + variation)
            
            trend_points.append({
                "date": point_date,
                "demand_index": round(point_demand, 1)
            })
        
        # Add skill-specific insights if skills provided
        skill_insights = []
        if skills:
            hot_skills = ["python", "react", "aws", "kubernetes", "tensorflow", "javascript", 
                          "devops", "docker", "machine learning", "sql", "agile"]
            
            for skill in skills:
                skill_lower = skill.lower()
                # Determine if this is a hot skill
                is_hot = any(hot in skill_lower for hot in hot_skills)
                growth = random.uniform(0.1, 0.25) if is_hot else random.uniform(-0.05, 0.15)
                
                skill_insights.append({
                    "skill": skill,
                    "demand_growth": round(growth, 2),
                    "popularity_rank": random.randint(1, 100),
                    "is_emerging": random.random() > 0.8
                })
        
        return {
            "job_title": job_title,
            "market_data": {
                "demand_growth_rate": round(growth_rate, 2),
                "demand_trend": trend_points,
                "job_postings_last_period": random.randint(1000, 50000),
                "average_time_to_fill": random.randint(20, 60),
                "competition_level": random.choice(["low", "medium", "high", "very high"])
            },
            "skill_insights": skill_insights,
            "regional_insights": {
                "top_locations": [
                    "San Francisco Bay Area",
                    "New York Metro", 
                    "Boston",
                    "Seattle",
                    "Austin"
                ],
                "remote_percentage": random.randint(20, 60)
            },
            "reference_period": time_period,
            "metadata": {
                "source": "RecruitX Simulated Market Data",
                "location": location or "United States (Average)",
                "analysis_date": datetime.now().strftime("%Y-%m-%d")
            }
        }
    
    def _simulate_skill_trends(
        self,
        skills: List[str],
        location: Optional[str],
        time_period: str
    ) -> Dict[str, Any]:
        """Generates realistic simulated skill trend data"""
        import random
        
        skill_trends = []
        
        # Growth factors for different skill categories
        growth_factors = {
            "ai": (0.2, 0.4),
            "machine learning": (0.2, 0.35),
            "data": (0.15, 0.3),
            "cloud": (0.15, 0.25),
            "frontend": (0.1, 0.2),
            "backend": (0.1, 0.2),
            "mobile": (0.05, 0.15),
            "devops": (0.15, 0.25),
            "security": (0.15, 0.3),
            "blockchain": (0.1, 0.35),
            "ar": (0.15, 0.3),
            "vr": (0.15, 0.3),
        }
        
        # Default growth for other categories
        default_growth = (0.05, 0.15)
        
        for skill in skills:
            skill_lower = skill.lower()
            
            # Determine growth factor based on skill category
            growth_range = default_growth
            for category, factor_range in growth_factors.items():
                if category in skill_lower:
                    growth_range = factor_range
                    break
            
            # Generate random growth within the appropriate range
            growth_rate = random.uniform(growth_range[0], growth_range[1])
            
            # Generate trend data points
            trend_points = []
            current_time = datetime.now()
            
            # Determine number of data points based on time period
            if time_period == "1month":
                num_points = 4  # Weekly
            elif time_period == "3months":
                num_points = 12  # Weekly
            elif time_period == "6months":
                num_points = 6  # Monthly
            else:  # 1year
                num_points = 12  # Monthly
                
            # Starting demand index, varied slightly for each skill
            base_demand = random.uniform(90, 110)
            
            for i in range(num_points):
                # Calculate date for this point
                if time_period in ["1month", "3months"]:
                    point_date = (current_time - timedelta(days=(num_points - i) * 7)).strftime("%Y-%m-%d")
                else:
                    point_date = (current_time - timedelta(days=(num_points - i) * 30)).strftime("%Y-%m-%d")
                    
                # Calculate demand with some randomness for realism
                point_growth = growth_rate * (i / num_points)  # Progressive growth
                variation = random.uniform(-0.05, 0.05)  # +/- 5% random variation
                point_demand = base_demand * (1 + point_growth + variation)
                
                trend_points.append({
                    "date": point_date,
                    "demand_index": round(point_demand, 1)
                })
            
            # Generate additional insights for each skill
            skill_trends.append({
                "skill": skill,
                "demand_growth_rate": round(growth_rate, 2),
                "demand_trend": trend_points,
                "popularity_rank": random.randint(1, 200),
                "job_postings_requiring": random.randint(500, 20000),
                "average_salary_impact": f"+{random.randint(5, 25)}%",
                "is_emerging": random.random() > 0.7
            })
        
        return {
            "skill_trends": skill_trends,
            "industry_insights": {
                "fastest_growing_industries": [
                    "Healthcare Technology",
                    "Fintech",
                    "Cybersecurity", 
                    "Clean Energy",
                    "E-commerce"
                ],
                "remote_work_trend": f"{random.randint(35, 75)}% of jobs in these fields offer remote options"
            },
            "reference_period": time_period,
            "metadata": {
                "source": "RecruitX Simulated Skill Trends",
                "location": location or "United States (Average)",
                "analysis_date": datetime.now().strftime("%Y-%m-%d"),
                "skills_analyzed": skills
            }
        }

# Create singleton instance
external_tool_service = ExternalToolService() 