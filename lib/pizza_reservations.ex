defmodule PizzaReservations do
  def predict(reservation_count, weight) do
    reservation_count * weight
  end

  def train(reservations, pizzas, iterations, learning_rate) do
    Enum.reduce_while(1..iterations, 0, fn _each, acc ->
      current_loss = loss(reservations, pizzas, acc)

      if loss(reservations, pizzas, acc + learning_rate) < current_loss do
        {:cont, acc + learning_rate}
      else
        {:halt, acc |> Float.round(2)}
      end
    end)
  end

  defp loss(reservations, pizzas, weight) do
    sum_of_losses(reservations, pizzas, weight) / length(reservations)
  end

  defp predictions(reservations, weight) do
    reservations
    |> Enum.map(&predict(&1, weight))
  end

  defp sum_of_losses(reservations, pizzas, weight) do
    predictions(reservations, weight)
    |> Enum.zip(pizzas)
    |> Enum.map(fn {prediction, actual} -> (prediction - actual) |> :math.pow(2) end)
    |> Enum.sum()
  end
end
