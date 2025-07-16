% Algomerative algorithm


% Load dataset
filename = 'wine_dataset_for_hiearchical_clusterig.csv';
data = readtable(filename);
X = zscore(table2array(data));  % Normalize data

% Pairwise distances
distances = pdist(X, 'euclidean');

% Linkage (agglomerative)
%Z = linkage(distances, 'ward');
%Z = linkage(distances,"single");
Z = linkage(distances, 'ward');

% Dendrogram
figure;
dendrogram(Z);
title('Agglomerative Hierarchical Clustering (Ward)');
xlabel('Sample Index');
ylabel('Distance');

% Cophenetic correlation coefficient
cophCorr = cophenet(Z, distances);
fprintf('Cophenetic Correlation Coefficient: %.4f\n', cophCorr);

% Create cluster labels (example: 3 clusters)
numClusters = 3;
clusterLabels = cluster(Z, 'maxclust', numClusters);

% Silhouette score
figure;
silhouette(X, clusterLabels);
title('Silhouette Plot - Agglomerative Clustering');

avgSilhouette = mean(silhouette(X, clusterLabels));
fprintf('Average Silhouette Score: %.4f\n', avgSilhouette);