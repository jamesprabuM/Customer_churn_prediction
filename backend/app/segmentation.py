import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any
import logging
import json

logger = logging.getLogger(__name__)

class ProfileSegmenter:
    """
    Applies strict KMeans clustering on meaningful customer behavioral features
    to assign risk segments (High Risk, Medium Risk, Low Risk) based on 3 clusters.
    """
    
    def __init__(self, k_clusters=3):
        self.k_clusters = 3  # Hardcoded to 3 as requested
        self.kmeans = KMeans(n_clusters=self.k_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.features_for_clustering = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.cluster_names: Dict[int, str] = {}
        self.is_fitted = False
        
    def fit_predict(self, df_raw: pd.DataFrame) -> pd.Series:
        """
        Builds the baseline clustering profiles from historical data
        and assigns risk labels based on tenure averages.
        """
        logger.info(f"Fitting KMeans with {self.k_clusters} clusters...")
        
        missing = [f for f in self.features_for_clustering if f not in df_raw.columns]
        if missing:
            raise KeyError(f"Missing essential clustering features: {missing}")
            
        # Select and scale
        cluster_data = df_raw[self.features_for_clustering].copy()
        
        # Coerce total charges and fill missing values
        cluster_data['TotalCharges'] = pd.to_numeric(cluster_data['TotalCharges'], errors='coerce').fillna(0)
        cluster_data['tenure'] = cluster_data['tenure'].fillna(0)
        cluster_data['MonthlyCharges'] = cluster_data['MonthlyCharges'].fillna(0)
        
        X_scaled = self.scaler.fit_transform(cluster_data)
        predictions = self.kmeans.fit_predict(X_scaled)
        cluster_data['Cluster'] = predictions
        
        # Analyze clusters to name personas based on feature values
        # We will use average tenure as a primary indicator of risk (lower tenure = higher risk)
        means = cluster_data.groupby('Cluster')['tenure'].mean().sort_values()
        
        # means.index gives the cluster IDs sorted by their average tenure (lowest to highest)
        sorted_clusters = means.index.tolist()
        
        if len(sorted_clusters) == 3:
            self.cluster_names[sorted_clusters[0]] = "High Risk"
            self.cluster_names[sorted_clusters[1]] = "Medium Risk"
            self.cluster_names[sorted_clusters[2]] = "Low Risk"
            
        self.is_fitted = True
        logger.info(f"Clustering complete. Identified risk segments: {json.dumps(self.cluster_names)}")
        return predictions

    def get_segment(self, row: pd.DataFrame) -> Dict[str, Any]:
        """
        Assigns a new customer to the correct risk segment and returns recommended action.
        """
        if not self.is_fitted:
            raise ValueError("KMeans model not fitted. Call fit_predict first.")
            
        # Parse inputs
        inputs = row[self.features_for_clustering].copy()
        inputs['TotalCharges'] = pd.to_numeric(inputs['TotalCharges'], errors='coerce').fillna(0)
        inputs['tenure'] = inputs['tenure'].fillna(0)
        inputs['MonthlyCharges'] = inputs['MonthlyCharges'].fillna(0)
        
        inputs_scaled = self.scaler.transform(inputs)
        cluster_id = self.kmeans.predict(inputs_scaled)[0]
        
        segment_name = self.cluster_names.get(int(cluster_id), "Unknown Risk")
        
        # Assign Action
        if segment_name == "High Risk":
            action = "Immediate intervention required. Offer maximum discount (₹500)."
        elif segment_name == "Medium Risk":
            action = "Monitor closely. Send targeted engagement email."
        elif segment_name == "Low Risk":
            action = "No immediate action needed. Maintain standard communication."
        else:
            action = "Review account history manually."
        
        return {
            "cluster_id": int(cluster_id),
            "segment": segment_name,
            "recommended_action": action,
            "features": inputs.iloc[0].to_dict()
        }
