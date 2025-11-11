from dash import html
import dash_bootstrap_components as dbc

def layout():
    return dbc.Container(
        dbc.Card(
            dbc.CardBody([
                html.H3("Introduction", style={"color":"#003662","marginBottom":"8px"}),
                html.P("Overview of the Skilled Labor Monitor and how to use each page."),
                html.Ul([
                    html.Li("Interactive Map to explore regional employment."),
                    html.Li("Sankey view for composition insights."),
                    html.Li("Tables for sortable details."),
                ], className="mb-0"),
            ]),
            style={"borderRadius":"20px","boxShadow":"0 4px 20px rgba(0,0,0,.10)","backgroundColor":"#ffffff"},
            className="my-3"
        ),
        fluid=True
    )
