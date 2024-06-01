from urllib.parse import urlparse
from socket import gethostbyname
import socket
import whoisdomain


class HostFeatures:

    def __init__(self, url):
        self.url = url
        self.urlparse = urlparse(self.url)

    def get_ip_address(self):
        try:
            domain = self.extract_domain()
            ip_address = gethostbyname(domain)
            return ip_address
        except socket.error as err:
            print("Have err")
            return None

    def extract_domain(self):
        domain_parts = self.urlparse.netloc.split('.')
        if len(domain_parts) > 1:  # Check if there's at least one subdomain
            # Return the last two parts joined by a dot
            return ".".join(domain_parts[-2:])
        else:
            return domain_parts[0]

    def get_whoisdomain(self):
        try:
            domain = self.extract_domain()
            d = whoisdomain.query(domain)
            attr = ['registrar', 'registrant_country',
                    'creation_date', 'expiration_date', 'last_updated', 'name_servers', 'registrant', 'emails']
            extracted_data = {key: d.__dict__[key]
                              for key in attr if key in d.__dict__}
            print(extracted_data)
        except Exception as e:
            print(e)

