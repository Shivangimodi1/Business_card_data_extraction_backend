const mysql = require('mysql2');

const db = mysql.createConnection({
    host: 'localhost',
    user: 'root',         // Default MySQL root user
    password: 'mysql@8.0.35',         // Your MySQL root password
    database: 'business_card_data'  // The database we created
});

// Test the connection
db.connect((err) => {
    if (err) {
        console.error('Database connection failed:', err.stack);
        return;
    }
    console.log('Connected to MySQL database.');
});

module.exports = db;
