import click


@click.group()
def cli():
    """An interactive visualization of the 2d O(2) model."""
    pass


@cli.command()
@click.option("-l", "--length", default=100, help="Size of the model")
@click.option("-b", "--beta", default=1.1199, help="Temperature variable")
def interact(length, beta):
    """Simple visualisation of the 2d O(2) model with interaction."""
    import O2py
    import matplotlib.pyplot as plt

    plot = O2py.interactiveo2plot(l=length, beta=beta)

    print("Interactive plot started, see plot window ...")
    while True:
        plt.pause(2)


if __name__ == "__main__":
    cli()
