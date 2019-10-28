
#include <iostream>
#include <any>
#include <functional>
#include <vector>


template <typename T>
T add(T x) { return x; }

template <typename T, typename ...Ts>
T add(T head, Ts... tail) { return head + add(tail...); }

// And what?

template <typename T>
std::any adder(std::any v1) {
    return std::function<std::any(std::any)>(
        [v1] (std::any v2) { return std::any(std::any_cast<T>(v1) + std::any_cast<T>(v2)); }
    );   
}

std::any apply_func(std::function<std::any(std::any)> func, const std::vector<std::any> &v)
{
    std::any f;
    size_t i = 0;
    for (; i < v.size()-1; i++) {
        func = std::any_cast<decltype(func)>(func(v[i]));
    }
    return func(v[i]);
}

// Even if it will be possible to combine variadic template lambda and curring,
// still required to store functions (ok, possibly in std::any),
// and still required to call them with proper number of args,
// and calling with variadic number of arguments can't be implemented in runtime.

int main()
{
    std::cout << add(1, 2, 3, 4, 5) << std::endl;
    
    std::vector<std::any> v = {5, 6};
    auto adder_i = adder<int>;
    auto result = apply_func(adder_i, v);
    std::cout << std::any_cast<int>(result) << std::endl;
    
    return 0;
}
