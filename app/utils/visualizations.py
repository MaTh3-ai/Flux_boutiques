import plotly.graph_objects as go

def plot_forecast(forecast_df, hist_n1, hist_n2, current_year):
    fig = go.Figure()

    # Dates = vraie colonne Date (pas l’index 0,1,2…)
    x_dates = forecast_df["Date"]

    # Prévision
    fig.add_trace(go.Scatter(
        x=x_dates, y=forecast_df["Prévision"],
        mode="lines+markers", name="Prévision", line=dict(color="blue")
    ))

    # Intervalle de confiance
    fig.add_trace(go.Scatter(
        x=list(x_dates) + list(x_dates[::-1]),
        y=list(forecast_df["Borne supérieure"]) + list(forecast_df["Borne inférieure"][::-1]),
        fill="toself", fillcolor="rgba(173,216,230,0.3)", name="Intervalle de confiance",
        line=dict(color="rgba(173,216,230,0.0)")
    ))

    # Historiques N‑1 / N‑2
    fig.add_trace(go.Scatter(
        x=x_dates, y=hist_n1, mode="lines", name=f"Année {current_year-1}",
        line=dict(dash="dash", color="green")
    ))
    fig.add_trace(go.Scatter(
        x=x_dates, y=hist_n2, mode="lines", name=f"Année {current_year-2}",
        line=dict(dash="dot",  color="orange")
    ))

    fig.update_layout(
        title="Prévisions hebdomadaires avec Historiques",
        xaxis_title="Date", yaxis_title="Valeur",
        hovermode="x unified", template="plotly_white"
    )
    return fig


def plot_historical_data(historical_full, cible):
    # Cette fonction trace l'intégralité des données historiques (agrégées par semaine)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
         x=historical_full.index,
         y=historical_full[cible],
         mode='lines',
         name='Historique complet',
         line=dict(color='red')
    ))
    fig.update_layout(title="Données historiques complètes",
                      xaxis_title="Date",
                      yaxis_title="Valeur",
                      template="plotly_white")
    return fig
