from invoke import task
import os


@task
def python(ctx):
    """ """
    ctx.run("which python" if os.name != "nt" else "where python")


@task
def git(ctx, message):
    ctx.run(f"git add .")
    ctx.run(f"git commit -m '{message}'")
    ctx.run(f"git push")


@task
def pull_data(ctx):
    ctx.run("dvc pull")


@task(pull_data)
def train(ctx):
    ctx.run("my_cli train")


@task
def dvc(ctx, folder="data", message="Add new data"):
    ctx.run(f"dvc add {folder}")
    ctx.run(f"git add {folder}.dvc .gitignore")
    ctx.run(f"git commit -m '{message}'")
    ctx.run(f"git push")
    ctx.run(f"dvc push")
