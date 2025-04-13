import dash
from dash import dcc, html, Input, Output
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Clustering and Elbow Method Visualization"),

    # Slider for number of clusters
    html.Div([
        html.Label("Select number of clusters:"),
        dcc.Slider(
            id="num-clusters",
            min=1,
            max=20,
            step=1,
            value=10,
            marks={i: str(i) for i in range(1, 21)}
        ),
    ], style={"margin-top": "20px"}),

    # Graph for elbow method
    dcc.Graph(id="elbow-graph"),
    # Graph for KMeans clustering results
    dcc.Graph(id="kmeans-cluster-graph",
              style={'width': '90vw', 'height': '90vh'}),
    # Graph for KMedoids clustering results
    dcc.Graph(id="kmedoids-cluster-graph",
              style={'width': '90vw', 'height': '90vh'}),
])


@app.callback(
    Output("elbow-graph", "figure"),
    Output("kmeans-cluster-graph", "figure"),
    Output("kmedoids-cluster-graph", "figure"),
    Input("num-clusters", "value")
)
def update_graphs(num_clusters):
    # Load dataset
    digits = load_digits()
    data, labels = digits.data, digits.target

    pca = PCA(n_components=3)
    data = pca.fit_transform(data)

    wcss = wcss_calculation(data, max_clusters=20)
    optimal_k_elbow = estimate_optimal_k_elbow(wcss)
    kmeans_labels, kmeans_centroids = kmeans(data, n_clusters=num_clusters)
    kmedoids_labels, kmedoids_centroids = kmedoids(
        data, n_clusters=num_clusters)

    elbow_fig = create_elbow_graph(wcss, optimal_k_elbow)
    kmeans_fig = create_cluster_graph(
        data, labels, kmeans_labels, kmeans_centroids, "KMeans")
    kmedoids_fig = create_cluster_graph(
        data, labels, kmedoids_labels, kmedoids_centroids, "KMedoids")

    return elbow_fig, kmeans_fig, kmedoids_fig


def create_cluster_graph(data, labels, kmeans_labels, kmeans_centroids, graph_name):
    kmeans_fig = go.Figure()

    # 1. Data points with original labels
    data_points = go.Scatter3d(
        x=data[:, 0],
        y=data[:, 1],
        z=data[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=kmeans_labels.astype(int),  # Default color by KMeans cluster
            colorscale='Viridis',
            colorbar=dict(title='Cluster'),
        ),
        # Hover info shows true digit
        text=[f"Digit: {digit}" for digit in labels],
        hoverinfo='text',
        name="Data Points (Clustered)"
    )

    # 2. KMeans centroids (added as diamonds)
    centroids = go.Scatter3d(
        x=kmeans_centroids[:, 0],
        y=kmeans_centroids[:, 1],
        z=kmeans_centroids[:, 2],
        mode='markers',
        marker=dict(
            size=10,
            color='black',
            symbol='diamond'
        ),
        name='K-Means Centroids'
    )

    # Add both to the figure
    kmeans_fig.add_trace(data_points)
    kmeans_fig.add_trace(centroids)

    # Button functionality to toggle color
    buttons = [
        # Button to color by KMeans clusters
        dict(
            label=f"Color by {graph_name}",
            method="restyle",
            args=[{"coloraxis.colorbar.title": "Cluster",
                   "marker.color": kmeans_labels.astype(int)}],
        ),
        # Button to color by original labels (true digits)
        dict(
            label="Color by Digit",
            method="restyle",
            args=[{"coloraxis.colorbar.title": "Digit",
                   "marker.color": labels.astype(int)}],
        ),
    ]

    # Layout adjustments
    kmeans_fig.update_layout(
        title=f"3D Clustering of MNIST Digits with {graph_name}",
        scene=dict(
            xaxis=dict(title='PCA 1'),
            yaxis=dict(title='PCA 2'),
            zaxis=dict(title='PCA 3'),
            xaxis_title='PCA 1',
            yaxis_title='PCA 2',
            zaxis_title='PCA 3',
        ),
        updatemenus=[
            dict(
                type="buttons",
                x=0.1,
                y=-0.15,
                buttons=buttons,
                showactive=True
            )
        ],
        legend=dict(itemsizing='constant')
    )

    return kmeans_fig


def create_elbow_graph(wcss, optimal_k_elbow):
    elbow_fig = go.Figure()
    elbow_fig.add_trace(go.Scatter(
        x=list(range(1, 21)),
        y=wcss,
        mode='lines+markers',
        name='WCSS',
        line=dict(color='blue', width=2)
    ))
    elbow_fig.add_annotation(
        x=optimal_k_elbow,
        y=wcss[optimal_k_elbow - 1],
        text=f"Elbow point: {optimal_k_elbow}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(size=12, color="black")
    )
    elbow_fig.update_layout(title="Elbow Method for Optimal Clusters",
                            xaxis_title="Number of Clusters",
                            yaxis_title="WCSS")
    return elbow_fig


def wcss_calculation(data, max_clusters=12):
    wcss = []
    for i in range(1, max_clusters + 1):
        labels, centroids = kmeans(data, n_clusters=i)
        distances = np.sqrt(((data - centroids[labels])**2).sum(axis=1))
        wcss.append(distances.sum())
    return wcss


def estimate_optimal_k_elbow(wcss):
    derivatives = np.diff(wcss)
    second_derivatives = np.diff(derivatives)
    optimal_clusters = np.argmin(second_derivatives) + 2
    return optimal_clusters


def kmeans(data, n_clusters=10, max_iter=4, tolerance=1e-4):
    np.random.seed(42)
    indices = np.random.choice(data.shape[0], size=n_clusters, replace=False)
    centroids = data[indices]
    for _ in range(max_iter):
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        updated_centroids = np.array(
            [data[labels == i].mean(axis=0) for i in range(n_clusters)])
        # Check for convergence
        if np.all(np.abs(updated_centroids - centroids) < tolerance):
            break
        centroids = updated_centroids
    return labels, centroids


def kmedoids(data, n_clusters=10, max_iter=4, tolerance=1e-4):
    np.random.seed(42)
    indices = np.random.choice(data.shape[0], size=n_clusters, replace=False)
    medoids = data[indices]
    labels = np.zeros(data.shape[0], dtype=int)
    for _ in range(max_iter):
        # Assign each point to the nearest medoid
        distances = np.sqrt(((data - medoids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        # Update medoids
        new_medoids = np.copy(medoids)
        for i in range(n_clusters):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                medoid_index = np.argmin(
                    np.sum(np.sqrt(
                        ((cluster_points - cluster_points[:, np.newaxis])**2).sum(axis=2)), axis=1)
                )
                new_medoids[i] = cluster_points[medoid_index]

        if np.all(np.abs(new_medoids - medoids) < tolerance):
            break
        medoids = new_medoids

    return labels, medoids


if __name__ == "__main__":
    app.run(debug=True)
