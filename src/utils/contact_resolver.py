import os
import re
import logging
import threading
import phonenumbers
from typing import Dict, Any, Optional, List, Set

# Configure logging
logger = logging.getLogger(__name__)

# Flag to detect if we're running on macOS with Contacts framework
HAS_CONTACTS_FRAMEWORK = False
try:
    import contacts
    from Foundation import NSCompoundPredicate
    HAS_CONTACTS_FRAMEWORK = True
except ImportError:
    pass

class EnhancedContactResolver:
    """Enhanced contact resolution using macOS Contacts framework.
    
    This class provides improved contact resolution for various identifier types
    (phone numbers, emails, iMessage IDs) with advanced caching and formatting.
    """
    
    def __init__(self, db):
        """Initialize contact resolver with database connection.
        
        Args:
            db: MessagesDB instance for database access
        """
        self.db = db
        self.contact_cache = {}  # Cache for contact lookups
        self.cache_lock = threading.Lock()  # Thread safety for cache access
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        # Initialize contacts framework
        self.has_contacts_framework = HAS_CONTACTS_FRAMEWORK
        if self.has_contacts_framework:
            try:
                self.contact_store = contacts.CNContactStore.alloc().init()
                # Request access if needed
                if not self._check_contacts_access():
                    logger.warning("No access to Contacts. Please grant permission in System Settings.")
            except Exception as e:
                logger.error(f"Error initializing Contacts framework: {e}")
                self.has_contacts_framework = False
    
    def _check_contacts_access(self):
        """Check if we have access to Contacts framework."""
        try:
            authorization = self.contact_store.authorizationStatusForEntityType_(
                contacts.CNEntityTypeContacts
            )
            
            if authorization == contacts.CNAuthorizationStatusNotDetermined:
                # Request access
                result = []
                event = threading.Event()
                
                def completion_handler(granted, error):
                    result.append(granted)
                    event.set()
                
                self.contact_store.requestAccessForEntityType_completionHandler_(
                    contacts.CNEntityTypeContacts,
                    completion_handler
                )
                
                # Wait for completion handler to be called
                event.wait(timeout=5.0)  # 5 second timeout
                return result[0] if result else False
                
            return authorization == contacts.CNAuthorizationStatusAuthorized
        except Exception as e:
            logger.error(f"Error checking Contacts access: {e}")
            return False
    
    def _is_email(self, identifier):
        """Check if the identifier is an email address."""
        if not identifier:
            return False
        return bool(self.email_pattern.match(identifier))
    
    def _is_phone_number(self, identifier):
        """Check if the identifier is a phone number."""
        if not identifier:
            return False
        # Remove common non-digit characters
        digits_only = re.sub(r'[^0-9+]', '', identifier)
        # Basic heuristic: if it has more than 7 digits, treat as phone number
        return len(digits_only) >= 7 and not self._is_email(identifier)
    
    def _format_phone_number(self, phone_number):
        """Format a phone number for better display and matching."""
        if not phone_number:
            return None
        
        try:
            # Try to parse and normalize the phone number
            parsed_number = phonenumbers.parse(phone_number, "US")
            if phonenumbers.is_valid_number(parsed_number):
                # Get different formats to improve matching and display
                e164 = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.E164)
                international = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
                national = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.NATIONAL)
                
                return {
                    "original": phone_number,
                    "e164": e164,
                    "international": international,
                    "national": national,
                    "search_values": {phone_number, e164, international, national, 
                                    e164.replace("+1", ""), 
                                    re.sub(r'[^0-9+]', '', phone_number)}
                }
        except Exception as e:
            logger.debug(f"Error formatting phone number {phone_number}: {e}")
        
        # If parsing fails, just return the original with a basic cleanup
        return {
            "original": phone_number,
            "e164": phone_number,
            "international": phone_number,
            "national": phone_number,
            "search_values": {phone_number, re.sub(r'[^0-9+]', '', phone_number)}
        }
    
    def _lookup_contact_by_phone(self, phone_number):
        """Look up a contact by phone number in macOS Contacts."""
        if not self.has_contacts_framework or not phone_number:
            return None
        
        formatted = self._format_phone_number(phone_number)
        if not formatted:
            return None
        
        try:
            # Create search predicate for phone number
            search_values = formatted["search_values"]
            predicates = []
            
            for value in search_values:
                # Skip empty values
                if not value:
                    continue
                # Create predicate for this phone value
                predicate = contacts.CNPhoneNumber.predicateForPhoneNumberMatchingStringWithLabels_(
                    value, None
                )
                predicates.append(predicate)
            
            if not predicates:
                return None
                
            # Combine with OR predicate
            final_predicate = NSCompoundPredicate.orPredicateWithSubpredicates_(predicates)
            
            # Setup keys to fetch
            keys = [
                contacts.CNContactGivenNameKey, 
                contacts.CNContactFamilyNameKey,
                contacts.CNContactNicknameKey,
                contacts.CNContactOrganizationNameKey,
                contacts.CNContactPhoneNumbersKey
            ]
            
            # Perform search
            matching_contacts = self.contact_store.unifiedContactsMatchingPredicate_keysToFetch_error_(
                final_predicate, keys, None
            )
            
            if matching_contacts and len(matching_contacts) > 0:
                # Process the first matching contact
                contact = matching_contacts[0]
                given_name = contact.givenName() or ""
                family_name = contact.familyName() or ""
                nickname = contact.nickname() or ""
                organization = contact.organizationName() or ""
                
                # Build name based on available parts
                contact_name = ""
                if given_name or family_name:
                    contact_name = f"{given_name} {family_name}".strip()
                elif nickname:
                    contact_name = nickname
                elif organization:
                    contact_name = organization
                
                if contact_name:
                    return contact_name
        except Exception as e:
            logger.debug(f"Error looking up contact by phone {phone_number}: {e}")
        
        return None
    
    def _lookup_contact_by_email(self, email):
        """Look up a contact by email address in macOS Contacts."""
        if not self.has_contacts_framework or not email:
            return None
        
        try:
            # Create search predicate for email
            predicate = contacts.CNContactEmailAddressesContainmentPredicate.predicateForEmailAddressWithStringValue_(email)
            
            # Setup keys to fetch
            keys = [
                contacts.CNContactGivenNameKey, 
                contacts.CNContactFamilyNameKey,
                contacts.CNContactNicknameKey,
                contacts.CNContactOrganizationNameKey,
                contacts.CNContactEmailAddressesKey
            ]
            
            # Perform search
            matching_contacts = self.contact_store.unifiedContactsMatchingPredicate_keysToFetch_error_(
                predicate, keys, None
            )
            
            if matching_contacts and len(matching_contacts) > 0:
                # Process the first matching contact
                contact = matching_contacts[0]
                given_name = contact.givenName() or ""
                family_name = contact.familyName() or ""
                nickname = contact.nickname() or ""
                organization = contact.organizationName() or ""
                
                # Build name based on available parts
                contact_name = ""
                if given_name or family_name:
                    contact_name = f"{given_name} {family_name}".strip()
                elif nickname:
                    contact_name = nickname
                elif organization:
                    contact_name = organization
                
                if contact_name:
                    return contact_name
        except Exception as e:
            logger.debug(f"Error looking up contact by email {email}: {e}")
        
        return None
    
    def resolve_contact(self, identifier):
        """Resolve any type of contact identifier to a name.
        
        This function will attempt to resolve phone numbers, email addresses,
        and iMessage IDs to human-readable names using macOS Contacts.
        
        Args:
            identifier: The phone number, email, or iMessage ID to resolve
            
        Returns:
            Dictionary with contact information or None if not found
        """
        if not identifier:
            return None
        
        # Check cache first
        with self.cache_lock:
            if identifier in self.contact_cache:
                return self.contact_cache[identifier]
        
        # Determine identifier type and lookup method
        contact_name = None
        identifier_type = "unknown"
        display_format = identifier
        
        if self._is_phone_number(identifier):
            # Handle phone number
            identifier_type = "phone"
            formatted = self._format_phone_number(identifier)
            if formatted:
                display_format = formatted["international"]
                contact_name = self._lookup_contact_by_phone(identifier)
        elif self._is_email(identifier):
            # Handle email address
            identifier_type = "email"
            display_format = identifier
            contact_name = self._lookup_contact_by_email(identifier)
        else:
            # Handle other iMessage IDs
            identifier_type = "imessage"
            display_format = identifier
            # Try both phone and email lookups
            contact_name = self._lookup_contact_by_phone(identifier) or self._lookup_contact_by_email(identifier)
        
        # Check if we found a contact
        if not contact_name:
            # If no contact found and looks like a phone number, try basic formatting
            if identifier_type == "phone":
                formatted = self._format_phone_number(identifier)
                if formatted:
                    display_format = formatted["international"]
        
        # Create result
        result = {
            "identifier": identifier,
            "type": identifier_type,
            "name": contact_name,
            "display_format": display_format,
            "display_name": self.format_display_name(identifier, contact_name, display_format)
        }
        
        # Cache the result
        with self.cache_lock:
            self.contact_cache[identifier] = result
        
        return result
    
    def format_display_name(self, identifier, contact_name=None, display_format=None):
        """Format a display name that includes both identifier and contact name.
        
        Args:
            identifier: Original identifier (phone, email, etc.)
            contact_name: Resolved contact name (if available)
            display_format: Formatted version of identifier
            
        Returns:
            Formatted display name string
        """
        if not identifier:
            return "Unknown"
        
        # If no display format provided, use identifier
        if not display_format:
            display_format = identifier
        
        # If no contact name or matches identifier, just return the formatted identifier
        if not contact_name or contact_name == identifier:
            return display_format
        
        # Check if the identifier is contained in the name (avoid redundancy)
        if identifier in contact_name or (display_format and display_format in contact_name):
            return contact_name
        
        # Return combined format: Name (identifier)
        return f"{contact_name} ({display_format})"
    
    def get_message_sender_info(self, message):
        """Get formatted sender information for a message.
        
        Args:
            message: Message dictionary from database
            
        Returns:
            Dictionary with sender information
        """
        is_from_me = message.get("is_from_me", False)
        
        if is_from_me:
            return {
                "identifier": "me",
                "type": "self",
                "name": "You",
                "display_format": "You",
                "display_name": "You",
                "is_self": True
            }
        
        # Get sender identifier
        sender_id = message.get("sender_id") or message.get("phone_number")
        if not sender_id:
            return {
                "identifier": "unknown",
                "type": "unknown",
                "name": None,
                "display_format": "Unknown",
                "display_name": "Unknown Sender",
                "is_self": False
            }
        
        # Resolve contact
        contact_info = self.resolve_contact(sender_id)
        contact_info["is_self"] = False
        return contact_info 