// marketplace.js

document.addEventListener("DOMContentLoaded", function() {
  console.log("Marketplace page loaded");

  // Animate product cards on load and hover
  const productCards = document.querySelectorAll('.product-card');
  
  productCards.forEach((card, index) => {
      // Staggered entrance animation
      card.style.animationDelay = `${index * 0.1}s`;

      card.addEventListener('mouseenter', () => {
          card.style.transform = 'translateY(-10px) scale(1.03)';
          card.style.transition = 'transform 0.3s ease';
      });

      card.addEventListener('mouseleave', () => {
          card.style.transform = 'translateY(0) scale(1)';
      });
  });

  // Add a simple filter feature
  const searchInput = document.createElement('input');
  searchInput.setAttribute('type', 'text');
  searchInput.setAttribute('id', 'searchInput');
  searchInput.setAttribute('placeholder', 'Search waste products...');
  searchInput.style.width = '100%';
  searchInput.style.maxWidth = '400px';
  searchInput.style.padding = '10px';
  searchInput.style.margin = '20px auto';
  searchInput.style.display = 'block';
  searchInput.style.borderRadius = '25px';
  searchInput.style.border = '1px solid #ccc';

  document.querySelector('.container').insertBefore(searchInput, document.getElementById('productGrid'));

  searchInput.addEventListener('input', function() {
      const filter = searchInput.value.toLowerCase();
      productCards.forEach(card => {
          const title = card.querySelector('.product-title').textContent.toLowerCase();
          const description = card.querySelector('.product-description').textContent.toLowerCase();
          if (title.includes(filter) || description.includes(filter)) {
              card.style.display = '';
          } else {
              card.style.display = 'none';
          }
      });
  });

  const productGrid = document.getElementById('productGrid');
  const cartItems = document.getElementById('cartItems');
  const cartTotal = document.getElementById('cartTotal');
  const checkoutBtn = document.getElementById('checkoutBtn');
  const filterSelect = document.getElementById('filterSelect');
  const sortSelect = document.getElementById('sortSelect');
  
  let cart = JSON.parse(localStorage.getItem('cart')) || [];
  let products = [];
  
  // Fetch products from the server
  async function fetchProducts() {
      try {
          const response = await fetch('/api/products');
          if (!response.ok) throw new Error('Failed to fetch products');
          
          products = await response.json();
          displayProducts(products);
          updateCartDisplay();
      } catch (error) {
          console.error('Error fetching products:', error);
          productGrid.innerHTML = '<div class="alert alert-danger">Failed to load products. Please try again later.</div>';
      }
  }
  
  // Display products in the grid
  function displayProducts(productsToShow) {
      productGrid.innerHTML = '';
      
      productsToShow.forEach(product => {
          const productCard = document.createElement('div');
          productCard.className = 'col-md-4 mb-4';
          productCard.innerHTML = `
              <div class="product-card">
                  <img src="${product.image}" class="product-image" alt="${product.title}">
                  <div class="product-info">
                      <h5 class="product-title">${product.title}</h5>
                      <p class="product-description">${product.description}</p>
                      <div class="d-flex justify-content-between align-items-center">
                          <span class="product-price">$${product.price.toFixed(2)}</span>
                          <button class="btn btn-primary add-to-cart" data-id="${product.id}">
                              Add to Cart
                          </button>
                      </div>
                  </div>
              </div>
          `;
          productGrid.appendChild(productCard);
      });
      
      // Add event listeners to the new buttons
      document.querySelectorAll('.add-to-cart').forEach(button => {
          button.addEventListener('click', addToCart);
      });
  }
  
  // Add product to cart
  function addToCart(event) {
      const productId = parseInt(event.target.dataset.id);
      const product = products.find(p => p.id === productId);
      
      if (!product) return;
      
      const existingItem = cart.find(item => item.id === productId);
      
      if (existingItem) {
          existingItem.quantity += 1;
      } else {
          cart.push({
              id: product.id,
              title: product.title,
              price: product.price,
              quantity: 1,
              image: product.image
          });
      }
      
      saveCart();
      updateCartDisplay();
      showNotification('Product added to cart!');
  }
  
  // Update cart display
  function updateCartDisplay() {
      cartItems.innerHTML = '';
      let total = 0;
      
      cart.forEach(item => {
          const cartItem = document.createElement('div');
          cartItem.className = 'cart-item';
          cartItem.innerHTML = `
              <div class="d-flex justify-content-between align-items-center">
                  <div>
                      <img src="${item.image}" alt="${item.title}" style="width: 50px; height: 50px; object-fit: cover;">
                      <span>${item.title}</span>
                  </div>
                  <div class="d-flex align-items-center">
                      <input type="number" class="form-control form-control-sm quantity-input" 
                             value="${item.quantity}" min="1" style="width: 60px;">
                      <span class="ms-2">$${(item.price * item.quantity).toFixed(2)}</span>
                      <button class="btn btn-sm btn-danger ms-2 remove-item" data-id="${item.id}">
                          <i class="fas fa-trash"></i>
                      </button>
                  </div>
              </div>
          `;
          cartItems.appendChild(cartItem);
          total += item.price * item.quantity;
      });
      
      cartTotal.textContent = `$${total.toFixed(2)}`;
      checkoutBtn.disabled = cart.length === 0;
      
      // Add event listeners to quantity inputs and remove buttons
      document.querySelectorAll('.quantity-input').forEach(input => {
          input.addEventListener('change', updateQuantity);
      });
      
      document.querySelectorAll('.remove-item').forEach(button => {
          button.addEventListener('click', removeFromCart);
      });
  }
  
  // Update item quantity
  function updateQuantity(event) {
      const productId = parseInt(event.target.closest('.cart-item').querySelector('.remove-item').dataset.id);
      const quantity = parseInt(event.target.value);
      
      if (quantity < 1) {
          event.target.value = 1;
          return;
      }
      
      const item = cart.find(item => item.id === productId);
      if (item) {
          item.quantity = quantity;
          saveCart();
          updateCartDisplay();
      }
  }
  
  // Remove item from cart
  function removeFromCart(event) {
      const productId = parseInt(event.target.closest('.remove-item').dataset.id);
      cart = cart.filter(item => item.id !== productId);
      saveCart();
      updateCartDisplay();
      showNotification('Product removed from cart');
  }
  
  // Save cart to localStorage
  function saveCart() {
      localStorage.setItem('cart', JSON.stringify(cart));
  }
  
  // Show notification
  function showNotification(message) {
      const notification = document.createElement('div');
      notification.className = 'notification';
      notification.textContent = message;
      document.body.appendChild(notification);
      
      setTimeout(() => {
          notification.remove();
      }, 3000);
  }
  
  // Handle search
  searchInput.addEventListener('input', () => {
      const searchTerm = searchInput.value.toLowerCase();
      const filteredProducts = products.filter(product => 
          product.title.toLowerCase().includes(searchTerm) ||
          product.description.toLowerCase().includes(searchTerm)
      );
      displayProducts(filteredProducts);
  });
  
  // Handle filtering
  filterSelect.addEventListener('change', () => {
      const filterValue = filterSelect.value;
      let filteredProducts = products;
      
      if (filterValue !== 'all') {
          filteredProducts = products.filter(product => 
              product.category === filterValue
          );
      }
      
      displayProducts(filteredProducts);
  });
  
  // Handle sorting
  sortSelect.addEventListener('change', () => {
      const sortValue = sortSelect.value;
      let sortedProducts = [...products];
      
      switch (sortValue) {
          case 'price-low':
              sortedProducts.sort((a, b) => a.price - b.price);
              break;
          case 'price-high':
              sortedProducts.sort((a, b) => b.price - a.price);
              break;
          case 'name':
              sortedProducts.sort((a, b) => a.title.localeCompare(b.title));
              break;
      }
      
      displayProducts(sortedProducts);
  });
  
  // Handle checkout
  checkoutBtn.addEventListener('click', async () => {
      try {
          const response = await fetch('/api/checkout', {
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json'
              },
              body: JSON.stringify({
                  items: cart,
                  total: cart.reduce((sum, item) => sum + (item.price * item.quantity), 0)
              })
          });
          
          if (!response.ok) throw new Error('Checkout failed');
          
          const result = await response.json();
          cart = [];
          saveCart();
          updateCartDisplay();
          showNotification('Order placed successfully!');
          
          // Redirect to order confirmation page
          window.location.href = `/order-confirmation/${result.order_id}`;
          
      } catch (error) {
          console.error('Error during checkout:', error);
          showNotification('Failed to place order. Please try again.');
      }
  });
  
  // Initial load
  fetchProducts();
});