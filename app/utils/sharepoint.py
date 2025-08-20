from io import BytesIO
import os
import joblib
from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.user_credential import UserCredential
from office365.sharepoint.files.file import File


def download_joblib(server_relative_url: str):
    """Download a joblib-serialized object from SharePoint.

    Parameters
    ----------
    server_relative_url: str
        Path of the file within the SharePoint site, starting with ``/sites``.
    """
    site_url = os.environ["SP_SITE"]
    user = os.environ.get("SP_USER")
    password = os.environ.get("SP_PASS")

    ctx = ClientContext(site_url).with_credentials(UserCredential(user, password))
    response = File.open_binary(ctx, server_relative_url)
    return joblib.load(BytesIO(response.content))
