import logging
import io
import asyncio
from typing import Optional, List, Dict, Any
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import pandas as pd
import gzip

logger = logging.getLogger(__name__)

class GoogleDriveAPI:
    def __init__(self, api_key: str):
        self.service = build("drive", "v3", developerKey=api_key)

    def find_subfolder(self, parent_id: str, name: str) -> Optional[str]:
        """Find a subfolder by name inside a parent folder"""
        # Removed mimeType from query to be less strict
        query = f"'{parent_id}' in parents and name = '{name}'"
        results = self.service.files().list(q=query, fields="files(id, name, mimeType)").execute() # Added mimeType to fields for debugging
        files = results.get("files", [])
        # Further filter by mimeType locally if needed, but return if found
        for f in files:
            if f['mimeType'] == 'application/vnd.google-apps.folder':
                return f["id"]
        return None

    def list_files_in_folder(self, folder_id: str) -> List[Dict[str, str]]:
        """List all files inside a folder"""
        results = self.service.files().list(
            q=f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder'",
            fields="files(id, name)"
        ).execute()
        return results.get("files", [])

    async def download_file_content(self, file_id: str) -> Optional[bytes]:
        """Download a file's content"""
        # Google Drive API doesn't have a direct async download.
        # We'll use a thread pool to run the blocking operation.
        loop = asyncio.get_event_loop()
        file_content = await loop.run_in_executor(
            None, self._perform_download, file_id
        )
        return file_content
    
    def _perform_download(self, file_id: str) -> Optional[bytes]:
        """Blocking download operation for use in executor"""
        try:
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                # logger.debug(f"Download progress: {int(status.progress() * 100)}%.")
            file_content = fh.getvalue()
            logger.debug(f"Downloaded content size for file {file_id}: {len(file_content)} bytes.")
            return file_content
        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {e}")
            return None
    
    def _read_csv_from_bytes(content: bytes) -> pd.DataFrame:
        """Reads CSV content from bytes, handling gzip compression.
        Returns an empty DataFrame if content is empty or cannot be read.
        """
        if not content:
            logger.debug("Received empty content for CSV reading.")
            return pd.DataFrame() # Return empty DataFrame for empty content
    
        try:
            # Try to decompress as gzip first
            with gzip.open(io.BytesIO(content), 'rt') as f:
                df = pd.read_csv(f)
                logger.debug(f"Successfully read {len(df)} rows from gzipped CSV.")
                return df
        except Exception as e:
            logger.debug(f"Failed to read as gzipped CSV ({e}). Trying as plain text.")
            try:
                # If not gzip, try reading directly as plain text
                df = pd.read_csv(io.BytesIO(content))
                logger.debug(f"Successfully read {len(df)} rows from plain CSV.")
                return df
            except Exception as e_plain:
                logger.warning(f"Failed to read CSV from bytes as plain text ({e_plain}). Returning empty DataFrame.")
                return pd.DataFrame()
    
    
    async def get_file_from_drive(
        gdrive_api: 'GoogleDriveAPI',
        root_folder_id: str,
        subfolder_names: List[str], # Change to list of subfolder names
        file_name_pattern: str
    ) -> Optional[bytes]:
        """
        Fetches a specific file from Google Drive based on a hierarchical folder structure.
        e.g., ROOT_FOLDER_ID/subfolder1/subfolder2/file_name_pattern
        """
        current_folder_id = root_folder_id
        full_path_str = "" # For logging purposes
    
        for subfolder_name in subfolder_names:
            full_path_str += f"/{subfolder_name}"
            found_id = gdrive_api.find_subfolder(current_folder_id, subfolder_name)
            if not found_id:
                logger.warning(f"Subfolder '{subfolder_name}' not found in path '{full_path_str}' under root '{root_folder_id}'.")
                return None
            current_folder_id = found_id
    
        # List files in the final folder and find the one matching the pattern
        files_in_folder = gdrive_api.list_files_in_folder(current_folder_id)
        target_file = next((f for f in files_in_folder if file_name_pattern in f['name']), None)
    
        if not target_file:
            logger.warning(f"File matching pattern '{file_name_pattern}' not found in '{full_path_str}' under root '{root_folder_id}'.")
            return None
    
        # Download file content
        return await gdrive_api.download_file_content(target_file['id'])