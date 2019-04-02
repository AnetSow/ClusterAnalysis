import kivy
kivy.require("1.10.1")

from kivy.app import App
import kivy.uix.gridlayout as TheGrid


class MyGrid(TheGrid.GridLayout):
    pass


class MyGridApp(App):

    def build(self):
        return MyGrid()


if __name__ == "__main__":
    MyGridApp().run()
