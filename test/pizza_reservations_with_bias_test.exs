defmodule PizzaReservationsWithBiasTest do
  use ExUnit.Case, async: true

  test "predict" do
    assert PizzaReservationsWithBias.predict(20, 2.1, 7) == 49
  end

  test "train" do
    {pizzas, reservations} = parse_file()

    {approximate_weight, approximate_bias} = PizzaReservationsWithBias.train(pizzas, reservations, 100_000, 0.01)

    assert approximate_weight == 1.1
    assert approximate_bias == 12.93
    assert_in_delta PizzaReservationsWithBias.predict(20, approximate_weight, approximate_bias), 34.93, 0.001
  end

  defp parse_file do
    File.read!("test/support/data/pizza_reservations.txt")
    |> String.split("\n", trim: true)
    |> Kernel.tl()
    |> Enum.map(fn each ->
      [reservations, pizzas] = String.split(each)
      {String.to_integer(reservations), String.to_integer(pizzas)}
    end)
    |> Enum.unzip()
  end
end
