"""
Function calling system for the PartSelect chat agent
"""
import re
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from optimized_scraping import PartSelectScraper, ScrapingResult, PartInfo
from config import config

logger = logging.getLogger(__name__)


@dataclass
class FunctionCall:
    """Represents a parsed function call"""
    function_name: str
    parameters: Dict[str, str]
    raw_match: str
    confidence: float = 1.0


@dataclass
class FunctionResult:
    """Result from function execution"""
    success: bool
    data: Any
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseFunctionHandler(ABC):
    """Base class for function handlers"""
    
    @property
    @abstractmethod
    def function_name(self) -> str:
        """Name of the function this handler manages"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this function does"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Dict[str, str]]:
        """Parameter definitions for this function"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> FunctionResult:
        """Execute the function with given parameters"""
        pass
    
    def validate_parameters(self, parameters: Dict[str, str]) -> Tuple[bool, str]:
        """Validate parameters for this function"""
        required_params = {k: v for k, v in self.parameters.items() 
                         if v.get('required', False)}
        
        missing_params = []
        for param_name in required_params:
            if param_name not in parameters or not parameters[param_name].strip():
                missing_params.append(param_name)
        
        if missing_params:
            return False, f"Missing required parameters: {', '.join(missing_params)}"
        
        return True, "Parameters valid"


class GetPartInformationHandler(BaseFunctionHandler):
    """Handler for get_part_information function"""
    
    def __init__(self, scraper: PartSelectScraper):
        self.scraper = scraper
    
    @property
    def function_name(self) -> str:
        return "get_part_information"
    
    @property
    def description(self) -> str:
        return "Get detailed information about a specific part or model number"
    
    @property
    def parameters(self) -> Dict[str, Dict[str, str]]:
        return {
            "model_number": {
                "type": "string",
                "description": "The part or model number to get information for",
                "required": True
            }
        }
    
    def execute(self, **kwargs) -> FunctionResult:
        """Execute get_part_information"""
        model_number = kwargs.get("model_number", "").strip()
        
        if not model_number:
            return FunctionResult(
                success=False,
                data=None,
                error_message="Model number is required"
            )
        
        try:
            logger.info(f"Getting part information for: {model_number}")
            scraping_result = self.scraper.get_part_information(model_number)
            
            if scraping_result.success:
                return FunctionResult(
                    success=True,
                    data=scraping_result,
                    metadata={
                        "url": scraping_result.url,
                        "part_title": scraping_result.part_info.title
                    }
                )
            else:
                return FunctionResult(
                    success=False,
                    data=scraping_result,
                    error_message=scraping_result.error_message or "Failed to retrieve part information"
                )
        
        except Exception as e:
            logger.error(f"Error in get_part_information: {e}")
            return FunctionResult(
                success=False,
                data=None,
                error_message=f"Unexpected error: {str(e)}"
            )


class CheckCompatibilityHandler(BaseFunctionHandler):
    """Handler for check_model_part_compatibility function"""
    
    def __init__(self, scraper: PartSelectScraper):
        self.scraper = scraper
    
    @property
    def function_name(self) -> str:
        return "check_model_part_compatibility"
    
    @property
    def description(self) -> str:
        return "Check if a specific part is compatible with a model"
    
    @property
    def parameters(self) -> Dict[str, Dict[str, str]]:
        return {
            "model_number": {
                "type": "string",
                "description": "The appliance model number",
                "required": True
            },
            "part_number": {
                "type": "string", 
                "description": "The part number to check compatibility for",
                "required": True
            }
        }
    
    def execute(self, **kwargs) -> FunctionResult:
        """Execute check_model_part_compatibility"""
        model_number = kwargs.get("model_number", "").strip()
        part_number = kwargs.get("part_number", "").strip()
        
        if not model_number or not part_number:
            missing = []
            if not model_number:
                missing.append("model_number")
            if not part_number:
                missing.append("part_number")
            
            return FunctionResult(
                success=False,
                data=None,
                error_message=f"Missing required parameters: {', '.join(missing)}"
            )
        
        try:
            logger.info(f"Checking compatibility: {model_number} + {part_number}")
            scraping_result = self.scraper.check_model_part_compatibility(model_number, part_number)
            
            if scraping_result.success:
                return FunctionResult(
                    success=True,
                    data=scraping_result,
                    metadata={
                        "url": scraping_result.url,
                        "model_number": model_number,
                        "part_number": part_number,
                        "compatibility": scraping_result.part_info.compatibility
                    }
                )
            else:
                return FunctionResult(
                    success=False,
                    data=scraping_result,
                    error_message=scraping_result.error_message or "Failed to check compatibility"
                )
        
        except Exception as e:
            logger.error(f"Error in check_model_part_compatibility: {e}")
            return FunctionResult(
                success=False,
                data=None,
                error_message=f"Unexpected error: {str(e)}"
            )


class FunctionCallParser:
    """Parses function calls from LLM responses"""
    
    # Enhanced patterns for function call detection
    FUNCTION_PATTERNS = [
        # Standard format: [function_name(param="value")]
        (r'\[get_part_information\(model_number="([^"]+)"\)\]', 
         'get_part_information', ['model_number']),
        
        # Compatibility check with two parameters
        (r'\[check_model_part_compatibility\(model_number="([^"]+)",\s*part_number="([^"]+)"\)\]', 
         'check_model_part_compatibility', ['model_number', 'part_number']),
        
        # Alternative formats with single quotes
        (r'\[get_part_information\(model_number=\'([^\']+)\'\)\]', 
         'get_part_information', ['model_number']),
        
        (r'\[check_model_part_compatibility\(model_number=\'([^\']+)\',\s*part_number=\'([^\']+)\'\)\]', 
         'check_model_part_compatibility', ['model_number', 'part_number']),
        
        # More flexible format without quotes
        (r'\[get_part_information\(model_number=([^\)]+)\)\]', 
         'get_part_information', ['model_number']),
        
        (r'\[check_model_part_compatibility\(model_number=([^,]+),\s*part_number=([^\)]+)\)\]', 
         'check_model_part_compatibility', ['model_number', 'part_number']),
    ]
    
    @classmethod
    def parse_function_calls(cls, text: str) -> List[FunctionCall]:
        """Parse all function calls from text"""
        function_calls = []
        
        for pattern, func_name, param_names in cls.FUNCTION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                parameters = {}
                for i, param_name in enumerate(param_names):
                    param_value = match.group(i + 1).strip().strip('"\'')
                    parameters[param_name] = param_value
                
                function_call = FunctionCall(
                    function_name=func_name,
                    parameters=parameters,
                    raw_match=match.group(0)
                )
                function_calls.append(function_call)
        
        return function_calls
    
    @classmethod
    def has_function_calls(cls, text: str) -> bool:
        """Check if text contains function calls"""
        return len(cls.parse_function_calls(text)) > 0


class FunctionManager:
    """Manages function handlers and execution"""
    
    def __init__(self, scraper: PartSelectScraper):
        self.scraper = scraper
        self.handlers: Dict[str, BaseFunctionHandler] = {}
        self._register_handlers()
    
    def _register_handlers(self) -> None:
        """Register all function handlers"""
        handlers = [
            GetPartInformationHandler(self.scraper),
            CheckCompatibilityHandler(self.scraper)
        ]
        
        for handler in handlers:
            self.handlers[handler.function_name] = handler
            logger.info(f"Registered function handler: {handler.function_name}")
    
    def get_available_functions(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available functions"""
        functions = {}
        for name, handler in self.handlers.items():
            functions[name] = {
                'name': handler.function_name,
                'description': handler.description,
                'parameters': handler.parameters
            }
        return functions
    
    def execute_function(self, function_call: FunctionCall) -> FunctionResult:
        """Execute a function call"""
        if function_call.function_name not in self.handlers:
            return FunctionResult(
                success=False,
                data=None,
                error_message=f"Unknown function: {function_call.function_name}"
            )
        
        handler = self.handlers[function_call.function_name]
        
        # Validate parameters
        is_valid, validation_message = handler.validate_parameters(function_call.parameters)
        if not is_valid:
            return FunctionResult(
                success=False,
                data=None,
                error_message=validation_message
            )
        
        # Execute function
        try:
            result = handler.execute(**function_call.parameters)
            result.metadata = result.metadata or {}
            result.metadata['function_name'] = function_call.function_name
            result.metadata['parameters'] = function_call.parameters
            return result
        
        except Exception as e:
            logger.error(f"Error executing function {function_call.function_name}: {e}")
            return FunctionResult(
                success=False,
                data=None,
                error_message=f"Function execution error: {str(e)}"
            )
    
    def process_response(self, response_text: str) -> Tuple[str, List[FunctionResult]]:
        """Process LLM response and execute any function calls"""
        function_calls = FunctionCallParser.parse_function_calls(response_text)
        
        if not function_calls:
            return response_text, []
        
        results = []
        processed_response = response_text
        
        for function_call in function_calls:
            logger.info(f"Executing function: {function_call.function_name} with params: {function_call.parameters}")
            
            result = self.execute_function(function_call)
            results.append(result)
            
            # Replace function call in response with result
            if result.success:
                replacement = self._format_function_result(result)
            else:
                replacement = f"Error: {result.error_message}"
            
            processed_response = processed_response.replace(function_call.raw_match, replacement)
        
        return processed_response, results
    
    def _format_function_result(self, result: FunctionResult) -> str:
        """Format function result for inclusion in response"""
        if not result.success:
            return f"Error: {result.error_message}"
        
        function_name = result.metadata.get('function_name', 'unknown')
        
        if function_name == 'get_part_information':
            return self._format_part_info_result(result.data)
        elif function_name == 'check_model_part_compatibility':
            return self._format_compatibility_result(result.data)
        else:
            return "Function executed successfully"
    
    def _format_part_info_result(self, scraping_result: ScrapingResult) -> str:
        """Format part information result"""
        if not scraping_result.success:
            return f"Sorry, I couldn't find information for that part/model number."
        
        info = scraping_result.part_info
        response_parts = []
        
        if info.title:
            response_parts.append(f"**{info.title}**")
        
        if info.partselect_number:
            response_parts.append(f"PartSelect Number: {info.partselect_number}")
        
        if info.manufacturer_part_number:
            response_parts.append(f"Manufacturer Part Number: {info.manufacturer_part_number}")
        
        if info.price:
            response_parts.append(f"Price: {info.price}")
        
        if info.availability:
            response_parts.append(f"Availability: {info.availability}")
        
        if info.description:
            response_parts.append(f"Description: {info.description}")
        
        if scraping_result.url:
            response_parts.append(f"More details: {scraping_result.url}")
        
        return "\n".join(response_parts)
    
    def _format_compatibility_result(self, scraping_result: ScrapingResult) -> str:
        """Format compatibility check result"""
        if not scraping_result.success:
            return "Sorry, I couldn't check compatibility for those part and model numbers."
        
        info = scraping_result.part_info
        model_number = scraping_result.url.split('/')[-2] if '/' in scraping_result.url else "the model"
        part_number = info.partselect_number or "the part"
        
        if info.compatibility == "Compatible" or scraping_result.success:
            response_parts = [
                f"Great news! Part {part_number} is compatible with model {model_number}."
            ]
            
            if info.title:
                response_parts.append(f"**{info.title}**")
            
            if info.price:
                response_parts.append(f"Price: {info.price}")
            
            if info.description:
                response_parts.append(f"Description: {info.description}")
            
            if scraping_result.url:
                response_parts.append(f"More details: {scraping_result.url}")
            
            response_parts.append("\nWould you like installation instructions for this part?")
        
        else:
            response_parts = [
                f"Part {part_number} does not appear to be compatible with model {model_number}.",
                "",
                "I recommend:",
                "• Double-checking the model number",
                "• Looking for alternative compatible parts", 
                "• Contacting customer service for assistance",
                "",
                "Would you like me to search for compatible alternatives?"
            ]
        
        return "\n".join(response_parts)