from pathlib import Path
import json

class Config:
    def __init__(self):
        self.config_dir = Path(__file__).parent.parent / 'config'
        self.data_config = self._load_config('data.json')
        self.fugle_config = self._load_config('fugle_marketdata.json')
        self.sinopac_config = self._load_config('sinopac.json')

    def _load_config(self, file_name: str) -> dict:
        """
        從 JSON 檔案載入設定。
        
        會先嘗試從主要設定檔載入，如果主要設定檔不存在則會改用範例設定檔。
        兩個檔案都必須放在 config 目錄下。
        
        Attributes:
            file_name: 設定檔名稱
            
        Returns:
            dict: 從 JSON 檔案載入的設定資料

        Notes:
        - 主要設定檔 (*.json) 不應該被提交到 git
        - 範例設定檔 (*.example.json) 作為設定範本
        - 請記得將 example.json 複製成 .json 並填入實際的設定值
        """

        main_path = self.config_dir / file_name
        example_path = self.config_dir / file_name.replace('.json', '.example.json')
        
        # Try loading from JSON files first
        if main_path.exists():
            with open(main_path, 'r') as f:
                return json.load(f)
        elif example_path.exists():
            with open(example_path, 'r') as f:
                return json.load(f)
        else:
            return {}

config = Config()
