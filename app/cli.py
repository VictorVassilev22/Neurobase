import click
# from app.map_reduce_custom_ask import run_query
from app.ask import run_query
from app.map_reduce_ask import run_query as run_map_reduce_query
from app.ingest import run_ingest


@click.group()
def cli():
    pass


@cli.command()
@click.option('--path', default='.', help='Path to codebase directory')
def ingest(path): # ingesting codebase into vectorstore
    """Ingest codebase into vectorstore"""
    run_ingest(path)

@cli.command()
@click.argument("question")
def ask(question):
    run_query(question)
    # run_map_reduce_query(question)

if __name__ == '__main__':
    cli()

